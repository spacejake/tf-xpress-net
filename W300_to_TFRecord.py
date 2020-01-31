from dataset.W300_parser import convert_TFRecords


if __name__ == '__main__':
    from utils import opts
    args = opts.argparser()
    convert_TFRecords(args)
