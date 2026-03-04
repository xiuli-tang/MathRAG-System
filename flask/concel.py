import re


def replace_single_dollar(content):
    """
    将单个 $ 包裹的内容替换为空格包裹的内容，而 $$ 包裹的内容保持不变。

    参数：
        content (str): 需要处理的文本

    返回：
        str: 处理后的文本
    """
    # 保护 $$...$$ 之间的内容，先提取出来
    protected_blocks = re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL)

    # 替换单个 $...$ 为空格包裹的内容
    modified_content = re.sub(r'\$(.*?)\$', r' \1 ', content)

    # 还原 $$...$$ 之间的内容
    for block in protected_blocks:
        modified_content = modified_content.replace(f' {block} ', f'$$ {block} $$', 1)

    return modified_content



