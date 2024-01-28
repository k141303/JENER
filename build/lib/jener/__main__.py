import re
import math
import argparse
import warnings

from collections import defaultdict

import torch

from omegaconf import OmegaConf

from liat_ml_roberta import RoBERTaTokenizer

from .model import load_finetuned_model
from .utils.array import slice, padding
from .utils.data import load_ene2name

warnings.simplefilter('ignore')

TEXT = "時事通信によると、社民党は国会の連立与党政権から離脱すると5月30日の全国幹事長会議で決定・発表した。これは米軍普天間基地の移設問題に関する対処に反対した福島みずほ党首が大臣を罷免（ひめん）されたことに社民党内の反発が大きく、地方組織の大半が連立から離脱するべきであると主張したことによる。読売新聞によると、福島さんは「全国幹事長会議で様々な意見を聞いて、その後常任幹事会を開いて政権からの離脱を決めた。全国の都道府県連の幹事長から『筋を通してよかった』といってもらった。離脱はきわめて残念で大きな決断だが、国民との信頼関係が全てだと考えた。米軍基地を沖縄県名護市（なごし）の辺野古（へのこ）地区に作らないと国民に約束してきた社民党としては、自分達の言葉に責任を持つ政治をやっていかなければいけない。社民党が国民に約束したことは変わっていない。変わったのは内閣の方だ。政権から離脱するので与党というわけには行かないが、法律によっては一緒に作ってきたものがあるから法案審議は是々非々でやっていく」と連立与党からの離脱について説明した。また民主党との選挙協力については、「選挙協力しているところもあるし、民主党として戦っているところも多い。今までも野党で共闘しながらの面もあった。社民党としての選挙を戦っていく」とした。また同じく読売によると、重野安正幹事長も「政権を離脱しても民主党・国民新党との政策合意は実現を求めていく。政権離脱=政策の協議もしないというわけではない。民主党の小沢一郎幹事長から福島さんの罷免後に電話があり、私（重野氏）からも『政権離脱後も選挙協力はしていただきたい』と話した」と語っている。"

def load_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="入力テキスト")

    parser.add_argument("--model_dir", type=str, default="./src/jener/data/model/jener_v1", help="JENERモデルの保存ディレクトリへのパス")
    parser.add_argument("--ene_def_path", type=str, default="./src/jener/data/ENE_Definition_v9.0.0-with-attributes-20220714.jsonl", help="ENE定義書へのパス")

    parser.add_argument("--seq_len", type=int, default=512, help="入力系列長")
    parser.add_argument("--dup_len", type=int, default=32, help="分割重複長")

    parser.add_argument("--deactive_cuda", action="store_true", help="GPUを使用しない場合に使用")

    return parser.parse_args()


def tokenize(text: str, tokenizer: RoBERTaTokenizer) -> (list, list):
    space_shift = {}
    for m in re.finditer("\s+", text):
        space_shift[m.span(0)[0]] = m.span(0)[1] - m.span(0)[0]

    tokens = tokenizer.tokenize(text)

    rm_head = lambda token: re.sub("@@$", "", token)
    orig_tokens = list(map(rm_head, tokens))

    offsets = []
    for token in orig_tokens:
        if len(offsets) == 0:
            s = 0
        else:
            s = offsets[-1][-1]
        s += space_shift.get(s, 0)
        offsets.append([s, s + len(token)])

    return tokens, offsets

def iob2offset(iob2_outputs: list, cfg: OmegaConf, model :torch.nn.Module) -> set:
    offsets = set()
    start = None
    iob2_tags = [model.set_labels[tag] for tag in iob2_outputs]
    if cfg.data.encoding in ["BIO"]:
        for i, tag in enumerate(iob2_tags):
            if start is not None and tag != "I-NE":
                offsets.add((start, i))
                start = None
            if tag == "B-NE":
                start = i
        if start is not None:
            offsets.add((start, len(iob2_outputs)))
    elif cfg.data.encoding in ["BIOUL"]:
        for i, tag in enumerate(iob2_tags):
            if tag == "I-NE":
                continue
            if start is not None:
                if tag == "L-NE":
                    offsets.add((start, i + 1))
                start = None
            if tag == "U-NE":
                offsets.add((i, i + 1))
            if tag == "B-NE":
                start = i
    else:
        raise NotImplementedError()

    return offsets

def convert_vec_to_ene_ids(vector: torch.Tensor, ene_tags: list) -> set():
    if (vector >= 0.5).any():
        vector = vector >= 0.5
        ene_ids = set(
            ene_tags[idx] for idx, flag in enumerate(vector.tolist()) if flag
        )
    else:
        ene_idx = vector.argmax(dim=-1).item()
        ene_ids = {ene_tags[ene_idx]}
    return ene_ids

def main():
    args = load_arg()

    cfg, model = load_finetuned_model(args.model_dir)
    model.eval()

    tokenizer = RoBERTaTokenizer.from_pretrained(cfg.model.bert.name)

    tokens, offsets = tokenize(args.input, tokenizer)

    spans = slice(len(tokens), args.seq_len-2, args.dup_len)

    outputs = {"iob2": torch.LongTensor(), "ene": torch.Tensor()}
    for i, (s, e) in enumerate(spans):
        input_tokens = [tokenizer.cls_token] + tokens[s:e] + [tokenizer.sep_token]
        attention_mask = [1] * len(input_tokens)

        input_tokens = padding(input_tokens, args.seq_len, tokenizer.pad_token)
        attention_mask = padding(attention_mask, args.seq_len, 0)

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

        input_ids = torch.LongTensor([input_ids])
        attention_mask = torch.BoolTensor([attention_mask])

        if not args.deactive_cuda and torch.cuda.is_available():
            model = model.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            outputs_ = model(input_ids, attention_mask)

        for key, value in outputs_.items():
            value = value.squeeze(0)
            value = value[:e-s+2][1:-1]
            if i != 0:
                value = value[:, math.ceil(args.dup_len):]
            if i != len(spans)-1:
                value = value[:, :-math.floor(args.dup_len)]

            outputs[key] = torch.cat([outputs[key], value.cpu()], dim=0)

    pred_offsets = iob2offset(outputs["iob2"].tolist(), cfg, model)

    ene2name = load_ene2name(args.ene_def_path)

    print(f"入力:{args.input}")
    for s, e in sorted(pred_offsets, key=lambda x: x[0]):
        s_, e_ = offsets[s][0], offsets[e-1][1]

        ene_ids = convert_vec_to_ene_ids(outputs["ene"][s], model.ene_tags)
        enes = [ene2name[ene_id] for ene_id in ene_ids]
        print(args.input[s_:e_], (s, e), enes)
        
        


if __name__ == "__main__":
    main()