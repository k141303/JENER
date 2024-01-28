import argparse
import warnings
warnings.simplefilter("ignore")

from .jener import JENER

def load_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="入力テキスト")

    parser.add_argument("--model_dir", type=str, default=None, help="JENERモデルの保存ディレクトリへのパス")
    parser.add_argument("--ene_def_path", type=str, default=None, help="ENE定義書へのパス")

    parser.add_argument("--seq_len", type=int, default=512, help="入力系列長")
    parser.add_argument("--dup_len", type=int, default=32, help="分割重複長")

    parser.add_argument("--deactive_cuda", action="store_true", help="GPUを使用しない場合に使用")

    return parser.parse_args()


def main():
    args = load_arg()
    jener = JENER(
        model_dir=args.model_dir, 
        ene_def_path=args.ene_def_path, 
        seq_len=args.seq_len, 
        dup_len=args.dup_len,
        deactive_cuda=args.deactive_cuda
    )
    predicts = jener(args.input)

    print("入力: {}".format(args.input))
    for d in predicts:
        print(d)


if __name__ == "__main__":
    main()