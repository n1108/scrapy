import numpy as np
import json
import os

# 配置路径
input_file = './process/wiki_knowledge.npy'
output_file = './process/wiki_knowledge_readable.json' # 输出文件名
LIMIT = 6  # 指定导出的条数，设置为 None 或 0 则导出全部

# 自定义 JSON 编码器
# 作用：把 Python 的 set (集合) 类型自动转换为 list (列表)，否则 JSON 无法保存
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def export_data(limit=None):
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        return

    print("正在加载数据，请稍候...")
    try:
        # 加载 .npy 数据
        data = np.load(input_file, allow_pickle=True)
        data_list = data.tolist() # 转为 Python 列表
        
        total_count = len(data_list)
        if limit and limit > 0:
            data_list = data_list[:limit]
            print(f"加载成功，原始数据共 {total_count} 条，将导出前 {len(data_list)} 条。")
        else:
            print(f"加载成功，共 {total_count} 条数据。")

        print("正在写入 JSON 文件...")

        # 写入 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                data_list, 
                f, 
                cls=SetEncoder,       # 使用自定义编码器处理 set
                indent=4,             # 缩进 4 格，保持美观
                ensure_ascii=False    # 关键：让中文直接显示，而不是显示 \uXXXX
            )

        print(f"✅ 成功导出 {len(data_list)} 条数据！")
        print(f"文件位置: {os.path.abspath(output_file)}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    export_data(limit=LIMIT)