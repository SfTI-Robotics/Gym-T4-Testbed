import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-network", help="select a network type")
parser.add_argument("-brain", help="select a brain type")
parser.add_argument("-preprocess", help="select a preprocess type")
args = parser.parse_args()


print(args.network)