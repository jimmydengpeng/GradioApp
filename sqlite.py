import sqlite3

# 数据库文件路径
database_path = 'audio.db'
# 要读取的表格名称
table_name = 't_20240408'
# 要读取的列名称
column_name = 'summary'

# 连接到SQLite数据库
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# 构造SQL查询语句
query = f'SELECT {column_name} FROM {table_name}'

# 执行查询
cursor.execute(query)

# 获取查询结果
results = cursor.fetchall()

# 打印结果
for row in results:
    print(row[0])

# 关闭游标和连接
cursor.close()
conn.close()