import numpy as np
import json

# 自定义一个 JSON 编码器，用来处理 set (集合) 类型
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)  # 将集合转换为列表，以便打印
        return super().default(obj)

# 文件路径
file_path = './process/wiki_knowledge.npy'

try:
    # 1. 加载数据
    data = np.load(file_path, allow_pickle=True)
    data_list = data.tolist()

    print(f"【统计】共加载了 {len(data_list)} 条数据。\n")

    # 2. 打印第一条数据
    if len(data_list) > 0:
        print("="*30 + " 第一条数据样例 " + "="*30)
        first_item = data_list[0]
        
        # 使用自定义的 cls=SetEncoder
        print(json.dumps(first_item, indent=4, ensure_ascii=False, cls=SetEncoder))
        
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
except Exception as e:
    print(f"读取出错：{e}")