"""
Author:zyq
Time:2021/10/8 13:18
"""
import argparse
parser=argparse.ArgumentParser(description="命令行中传入一个数字")
parser.add_argument('intergers',type=str,help="传入的数字")

args=parser.parse_args()

print(args)