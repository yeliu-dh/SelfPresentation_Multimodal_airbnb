import pandas as pd
import logging
import os, sys


def save_csv_as_latex(table_csv, output_path, caption, label, ndigits=4, 
                      escape=True, index=True):
    """
    Save a pandas DataFrame to LaTeX, safe for compilation.
    
    Args:
        table_csv (pd.DataFrame): 要保存的表格
        output_path (str): 输出路径
        caption (str): 表格标题
        label (str): 表格标签
        ndigits (int, optional): 保留小数位
        escape (bool): 是否转义 LaTeX 特殊字符
        index (bool): 是否保留行索引
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = table_csv.copy()
    # 空值处理
    df = df.fillna('-')
    # print(df.dtypes)
    
    # 将每个单元格换行的数值（比如 β \n (std)）用 makecell 包裹
    def wrap_cell(val):
        if isinstance(val, str) and '\n' in val:
            return r'\makecell{' + val.replace('\n','\\\\') + '}'
        return val
    df = df.applymap(wrap_cell)

    if ndigits :
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].round(ndigits)    
    
    # 生成 LaTeX
    latex_code = df.to_latex(
        caption=caption,
        label=label,
        index=index,
        escape=escape,       # True 转义特殊字符，False 保留原样
        longtable=False,
        multicolumn=True,
        multicolumn_format='c',
        bold_rows=False,
        float_format=lambda x: f"{x:.{ndigits}f}" if isinstance(x, (int, float)) and ndigits else x
        # 对于数值型显示前ndigits位
    )
    # print(latex_code)
    
    # 保存
    with open(output_path, 'w') as f:
        f.write(latex_code)
    
    print(f"☑️[SAVE] Table saved as LaTeX to {output_path}!\n")
    return #latex_code