import datetime

month_zh_to_int = {
    "一月": 1,
    "二月": 2,
    "三月": 3,
    "四月": 4,
    "五月": 5,
    "六月": 6,
    "七月": 7,
    "八月": 8,
    "九月": 9,
    "十月": 10,
    "十一月": 11,
    "十二月": 12
}

month_zh_to_str = {
    "一月": "01",
    "二月": "02",
    "三月": "03",
    "四月": "04",
    "五月": "05",
    "六月": "06",
    "七月": "07",
    "八月": "08",
    "九月": "09",
    "十月": "10",
    "十一月": "11",
    "十二月": "12"
}

num_to_month = {
    1: '一月',
    2: '二月',
    3: '三月',
    4: '四月',
    5: '五月',
    6: '六月',
    7: '七月',
    8: '八月',
    9: '九月',
    10: '十月',
    11: '十一月',
    12: '十二月'
}

all_months_zh = ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"]

def is_valid_year(year: int):
    current_year = datetime.datetime.now().year
    if isinstance(year, int) and year > 0 and year <= current_year:
        return True
    else:
        return False


def get_cur_month_zh():
    return num_to_month[int(datetime.datetime.now().month)]

def get_date_str(year, month: str, day=None) -> str:
    # TODO: refactor
    assert isinstance(month, str)
    return f"{year}{get_month_from_zh_to_str(month)}"

def get_month_from_zh_to_str(month: str) -> int:
    return month_zh_to_str[month]