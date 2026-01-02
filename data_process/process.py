import numpy as np
import os
import random as rd
import json
import re
from tqdm import tqdm
from lxml import etree
from langconv import *
from filter_words import filter_url

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子或列表
    :return: 转换后的结果
    '''
    if isinstance(sentence, str):
        return Converter('zh-hans').convert(sentence)
    elif isinstance(sentence, list):
        return [Converter('zh-hans').convert(str(i)) for i in sentence]
    elif str(type(sentence)) == "<class 'lxml.etree._ElementUnicodeResult'>":
        return Converter('zh-hans').convert(str(sentence))
    return sentence

def filter(entity_title, category_list):
    # entity_title:string, category_list:list
    # filter_url = ['游戏', '%E6%B8%B8%E6%88%8F', '维基', '%E7%BB%B4%E5%9F%BA', '幻想', '我的世界', '魔兽']
    for i in filter_url:
        if i in entity_title:
            return True
        for j in category_list:
            if i in j:
                return True
    return False


def unified_string(object):
    # 如果是一个列表，则转换为字符串，如果是字符串则直接返回
    if type(object) == list:
        return ''.join(object)
    if type(object) == str:
        return object

def extract_infobox(content):
    '''
    维基百科页面中的侧边栏（infobox 或 sidebar）包含结构化数据。
    :param content: etree node
    :return: dict
    '''
    # 查找 infobox 或 sidebar 类型的表格
    # 修复：只针对顶级 table 进行处理，避免嵌套 table 导致的重复提取
    tables = content.xpath(".//table[contains(@class,'infobox') or contains(@class,'sidebar')]")
    knowledge = dict()
    
    for table in tables:
        # 仅获取该 table 下直接相关的 tr，避免处理嵌套在 td 里的子 table 的 tr
        rows = table.xpath("./tbody/tr | ./tr")
        for row in rows:
            # 提取 th 作为键
            th_elements = row.xpath("./th")
            if not th_elements:
                continue
            
            # 合并所有 th 的文本作为键名
            th_text = " ".join(["".join(th.xpath(".//text()")).strip() for th in th_elements]).strip()
            th_text = th_text.replace('[编辑]', '').replace('[編輯]', '').strip()
            
            if not th_text:
                continue
            
            # 提取 td 中的文本
            td_elements = row.xpath("./td")
            if not td_elements:
                continue
                
            # 提取 td 文本，过滤掉多余空白，并移除参考文献
            td_text_list = []
            for td in td_elements:
                # 临时移除参考资料等噪声，以免混入结构化数据
                for noise in td.xpath(".//*[contains(@class, 'reference')]"):
                    parent = noise.getparent()
                    if parent is not None:
                        parent.remove(noise)
                
                text = " ".join(td.xpath(".//text()")).strip()
                if text:
                    td_text_list.append(text)
            
            td_text = " ".join(td_text_list).strip()
            
            if th_text and td_text:
                th_text = Traditional2Simplified(th_text)
                td_text = Traditional2Simplified(td_text)
                
                # 如果键已存在且内容不同，则追加；如果内容完全一样（可能是响应式设计的残留），则跳过
                if th_text in knowledge:
                    if td_text not in knowledge[th_text]:
                        knowledge[th_text] += " | " + td_text
                else:
                    knowledge[th_text] = td_text
    return knowledge

def extract_navbox(content):
    '''
    维基百科底部的导航框 (Navbox)，提取实体关联信息。
    :param content: etree node
    :return: list of dicts
    '''
    # 查找所有包含 navbox 类的表格（通常是 navbox-inner）
    navboxes = content.xpath(".//table[contains(@class,'navbox')]")
    knowledge = list()
    for i in navboxes:
        know = dict()
        groups = []
        
        # 1. 提取 Navbox 标题
        navbox_title_el = i.xpath(".//th[contains(@class, 'navbox-title')]")
        if not navbox_title_el:
            continue
            
        # 复制一份以防修改原树影响后续（虽然这里只是读）
        title_node = navbox_title_el[0]
        # 移除标题中的 "查 论 编" 导航条
        navbar = title_node.xpath(".//*[contains(@class, 'navbar')]")
        title_text = ""
        if navbar:
            # 如果有 navbar，提取非 navbar 部分的文本
            # 这是一个简单的技巧：获取所有文本，然后去掉 navbar 的文本
            all_text = "".join(title_node.xpath(".//text()")).strip()
            navbar_text = "".join(navbar[0].xpath(".//text()")).strip()
            title_text = all_text.replace(navbar_text, "").strip()
        else:
            title_text = "".join(title_node.xpath(".//text()")).strip()

        if not title_text:
            continue
            
        root = Traditional2Simplified(title_text)
        
        # 2. 提取 Navbox 内部的分组行
        navbox_tr = i.xpath(".//tr")
        for j in navbox_tr:
            group_th = j.xpath(".//th[contains(@class, 'navbox-group')]")
            list_td = j.xpath(".//td[contains(@class, 'navbox-list')]")
            
            if group_th and list_td:
                group_name = Traditional2Simplified("".join(group_th[0].xpath(".//text()")).strip())
                # 提取该分组下的所有链接实体，优先使用 title 属性
                items_nodes = list_td[0].xpath(".//a[not(contains(@class, 'external'))]")
                items = []
                for item_node in items_nodes:
                    # 优先取 title，如果没有则取 text
                    name = item_node.get('title') or "".join(item_node.xpath(".//text()")).strip()
                    if name:
                        # 移除“（页面不存在）”后缀
                        name = name.replace('（页面不存在）', '').replace('(页面不存在)', '').strip()
                        items.append(Traditional2Simplified(name))
                
                if items:
                    # 去重，保持顺序
                    seen = set()
                    unique_items = []
                    for item in items:
                        if item not in seen:
                            unique_items.append(item)
                            seen.add(item)
                    groups.append({group_name: unique_items})
        
        if groups:
            know[root] = groups
            knowledge.append(know)
    return knowledge

def extract_paragraph(content):
    # 抽取段落
    '''
    维基百科页面的主要内容为段落文本（部分会有插图，暂时忽略图片，对存在latex的公式则保存）；
    维基百科一开始是一个摘要，然后是目录，下面则是根据目录中的子标题分别展示相应的文本内容。我们只取标签为<h3>对应为子标题，p等作为文本
    :param paragraph:
    :return:
    '''

    def process_text(text_elements):
        # 处理段落中的文本和特殊元素（如公式）
        text_process = []
        for element in text_elements:
            if isinstance(element, str) or str(type(element)) == "<class 'lxml.etree._ElementUnicodeResult'>":
                cleaned = element.strip()
                if cleaned:
                    text_process.append(Traditional2Simplified(cleaned))
            elif hasattr(element, 'xpath'):
                # 处理 LaTeX 公式
                latex_imgs = element.xpath(".//img[@class='mwe-math-fallback-image-inline']/@alt | .//img[@class='mwe-math-fallback-image-display']/@alt")
                if latex_imgs:
                    # 清理 LaTeX 字符串，移除 \displaystyle 和外层花括号
                    latex_str = latex_imgs[0]
                    latex_str = latex_str.replace('{\\displaystyle ', '').replace('\\displaystyle ', '')
                    text_process.append('_latex_:' + latex_str.strip())
        
        # 改进：将碎片化的文本拼接起来
        # 如果相邻的片段完全相同，则只保留一个（处理 Wikipedia 内部可能存在的重复渲染文本）
        refined_process = []
        for i, text in enumerate(text_process):
            if i > 0 and text == text_process[i-1]:
                continue
            refined_process.append(text)
            
        return " ".join(refined_process).strip()

    # 适配新的维基百科结构
    paragraph_nodes = content.xpath("./p | ./h2 | ./h3 | ./ul | ./ol | ./dl | ./pre | ./div[contains(@class, 'mw-heading')]")
    passage = {'abstract': []} 
    sub_content = dict() 
    entities = [] 
    sub_title = '' 
    
    # 需要忽略的章节标题
    ignore_sections = ['参考文献', '外部链接', '参见', '注释', '参考资料', '相关条目', '扩展阅读', '资料来源', '外部连结', '外部连线']
    skip_current_section = False
    
    for i in paragraph_nodes:
        tag = i.tag
        current_node = i
        if tag == 'div' and i.get('class') and 'mw-heading' in i.get('class'):
            h_nodes = i.xpath("./h2 | ./h3")
            if h_nodes:
                current_node = h_nodes[0]
                tag = current_node.tag
            else:
                continue

        if tag in ['h2', 'h3']:
            title_text = ''.join(current_node.xpath(".//text()")).strip()
            title_text = title_text.replace('[编辑]', '').replace('[編輯]', '').strip()
            
            if not title_text:
                continue

            if any(ignore in title_text for ignore in ignore_sections):
                skip_current_section = True
                sub_title = ''
                continue
            
            sub_title = Traditional2Simplified(title_text)
            skip_current_section = False
            continue
            
        if skip_current_section:
            continue
            
        text_elements = i.xpath(".//text() | .//span[@class='mwe-math-element']")
        processed_text = process_text(text_elements)
        
        # 提取链接作为实体
        current_entities = i.xpath(".//a[not(contains(@class, 'external'))]/@title")
        for e in current_entities:
            if e:
                # 移除“（页面不存在）”后缀，使实体名更纯净
                clean_e = e.replace('（页面不存在）', '').replace('(页面不存在)', '').strip()
                entities.append(Traditional2Simplified(clean_e))

        if tag == 'pre':
            code_text = ''.join(i.xpath(".//text()"))
            processed_text = '_code_:' + code_text

        if not processed_text:
            continue

        # 进一步清理段落文本中的连续空格
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()

        if sub_title == '':
            passage['abstract'].append(processed_text)
        else:
            if sub_title not in sub_content:
                sub_content[sub_title] = []
            sub_content[sub_title].append(processed_text)
            
    passage['paragraphs'] = sub_content
    passage['entities'] = set(entities)
    return passage

def process_html(content_html):
    tree = etree.HTML(content_html)
    
    # 1. 移除 style, script 和 HTML 注释
    for el in tree.xpath(".//style | .//script | .//comment()"):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

    divs = tree.xpath("//div[contains(@class, 'mw-parser-output')]")
    
    if not divs:
        return {}, [], {'abstract': [], 'paragraphs': {}, 'entities': set()}

    all_infobox_know = {}
    all_navbox_know = []
    all_passage = {'abstract': [], 'paragraphs': {}, 'entities': set()}

    # 遍历所有 mw-parser-output 块，提取信息并合并
    for div in divs:
        # 1. 提取 infobox (必须在清理噪声标签之前提取)
        infobox_know = extract_infobox(div)
        all_infobox_know.update(infobox_know)

        # 2. 提取 navbox (必须在清理噪声标签之前提取)
        navbox_know = extract_navbox(div)
        all_navbox_know.extend(navbox_know)

        # 3. 清理噪声标签 (现在可以安全清理了，因为结构化数据已经提取)
        bad_classes = [
            'reflist', 'navbox', 'infobox', 'reference', 'mw-editsection', 
            'noprint', 'metadata', 'hatnote', 'sidebar', 'stub', 'alert',
            'sistersitebox', 'portal', 'authcontrol-content', 'nmbox', 'navbox-styles'
        ]
        for cls in bad_classes:
            for bad_el in div.xpath(f".//*[contains(@class, '{cls}')]"):
                parent = bad_el.getparent()
                if parent is not None:
                    parent.remove(bad_el)

        # 4. 提取段落
        passage = extract_paragraph(div)
        
        # 合并摘要
        all_passage['abstract'].extend(passage['abstract'])
        
        # 合并子章节内容
        for sub_title, content in passage['paragraphs'].items():
            if sub_title not in all_passage['paragraphs']:
                all_passage['paragraphs'][sub_title] = []
            all_passage['paragraphs'][sub_title].extend(content)
            
        # 合并实体
        all_passage['entities'].update(passage['entities'])

    return all_infobox_know, all_navbox_know, all_passage



def read_files(origin_page, save_path):
    # 读取所有处理的数据集
    if not os.path.isdir(origin_page):
        raise Exception("请给出合法的目录")
    wiki_knowledge = []
    if os.path.exists(save_path + 'wiki_knowledge.npy'):
        # wiki_knowledge = (np.load('wiki_knowledge.npy')[()]).tolist()
        pass
    files = os.listdir(origin_page)
    # files = ['快速排序.txt']
    num = 0
    for file in tqdm(files):
        if file[-4:] != '.txt':
            continue
        with open(origin_page + file, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
        entity_title = Traditional2Simplified(lines[0][3:].replace('\n', ''))
        category_list = Traditional2Simplified(lines[1][3:].replace('\n', '').split('\t'))

        if filter(entity_title, category_list): # 如果实体标题或分类中包含一些过滤词，则不再处理当前文本
            continue
        url = lines[2][5:].replace('\n', '')
        time = lines[3][5:].replace('\n', '')
        content = ''.join(lines[5:]).replace('\n', ' ')
        infobox_know, navbox_know, passage = process_html(content)
        knowledge = dict()
        knowledge['entity'] = entity_title
        knowledge['category'] = category_list
        knowledge['url'] = url
        knowledge['time'] = time
        knowledge['structure_know'] = infobox_know # 维基百科中的infobox最终定义为该实体的结构化知识
        knowledge['corrseponding_know'] = navbox_know # 维基百科中的navbox最终定义与该实体有关的实体的结构化知识
        knowledge['smi-structure_know'] = passage # 维基百科中的段落被定位为该实体的半结构化知识
        wiki_knowledge.append(knowledge)
        num += 1
        if num%500 == 0: # 每隔一段时间保存一次防止中途报错而导致前面的数据丧失
            np.save(save_path + "wiki_knowledge.npy", wiki_knowledge)
    np.save(save_path + "wiki_knowledge.npy", wiki_knowledge)
    print("已完成处理所有维基百科知识，总数量为{}".format(len(wiki_knowledge)))


if __name__ == '__main__':
    origin_page = './origin_page/'
    save_path = './process/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    read_files(origin_page, save_path)