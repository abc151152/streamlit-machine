# 读取文件并检测非标准字符
with open('D:\machine-streamlit\pages\random.py', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 检查每行内容，打印包含特殊字符的行
for i, line in enumerate(lines, 1):
    if '\u00A0' in line or any(ord(char) > 127 for char in line):  # 查找非 ASCII 字符
        print(f"Line {i}: {repr(line)}")

# 替换特殊字符
cleaned_lines = [line.replace('\u00A0', ' ') for line in lines]  # 替换为标准空格

# 将清理后的代码保存
with open('random.py', 'w', encoding='utf-8') as file:
    file.writelines(cleaned_lines)
print("非标准字符已替换并保存到新文件:random.py")
