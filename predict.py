import argparse


def main(dict_args):
    pass


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Predict app')
    argparser.add_argument('image_path', type=str)
    argparser.add_argument('checkpoint', type=str)
    argparser.add_argument('--top_k', type=int, default=5)
    argparser.add_argument('--gpu', action='store_true')
    args = argparser.parse_args()
    print(args)
    dict_args = vars(args)
    main(dict_args)
