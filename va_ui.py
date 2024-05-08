import os, shutil
import os.path as osp

import gradio as gr
from gradio_calendar import Calendar
import sqlite3
import torch
from transformers import pipeline
from langchain_community.llms import Ollama

from date_utils import *


MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000
VIDEO_LENGTH_LIMIT_S = 3600  # TODO(jimmy) limit to 1 hour video files

device = 0 if torch.cuda.is_available() else "cpu"


# --- 设置全局路径 --- #
# 音频文件所在路径
AUDIO_PTH = './audio'
AUDIO_DATE_LIST = os.listdir(AUDIO_PTH)

# 视频文件所在路径
ROOT_MEETING = './meeting'
MEETING_DATE_LIST = os.listdir(ROOT_MEETING)

# 数据库文件路径
DB_PTH_A = 'audio.db'
DB_PTH_M = 'meeting.db'

# 临时文件目录
TMP_PTH = './tmp/'
TMP_PTH_A = './tmp/audio'
TMP_PTH_M = './tmp/meeting'
for p in [TMP_PTH, TMP_PTH_A, TMP_PTH_M]:
    if not osp.exists(p): os.mkdir(p)

CUR_DATE_PTH_A = AUDIO_DATE_LIST[0] # e.g. 2024XXXX
CUR_DATE_FILTERED_M = MEETING_DATE_LIST[0] # e.g. 2024XX
IS_FILTERED_A = False
IS_FILTERED_M = False


TEST_LOAD_MODEL = False
if TEST_LOAD_MODEL:
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=device)
    llm = Ollama(model="qwen:7b")


# --- Button Click Listeners --- #

def btn_filter_a_listener(in_date, in_num, in_text):
    global IS_FILTERED_A
    IS_FILTERED_A = True
    return _filter_audio_files_in_db(in_date, in_num, in_text), _get_cancel_btn(IS_FILTERED_A)

def btn_filter_m_listener(in_year: int, in_month: str, in_keyword_m):
    date = get_date_str(in_year, in_month)
    # _filter_meeting_files_in_db(date, in_keyword_m)
    return _filter_meeting_files_in_db(date, in_keyword_m), _get_cancel_btn(IS_FILTERED_M)


def _get_fe_all(root: str):
    return gr.FileExplorer(
        glob="*.*", 
        file_count="single",
        root_dir=root,
        # ignore_glob=".*",
        label="所有文件")

def _get_cancel_btn(interactive):
    return gr.Button("清除", interactive=interactive)

def btn_all_file_listener():
    global IS_FILTERED_A
    IS_FILTERED_A = False
    return _get_fe_all(AUDIO_PTH)


def btn_all_file_m_listener():
    global IS_FILTERED_M
    IS_FILTERED_M = False
    return _get_fe_all(ROOT_MEETING)


def btn_filter_cancel_listener():
    global IS_FILTERED_A
    IS_FILTERED_A = False
    fe_all = _get_fe_all()
    in_cal = Calendar(value=None, type="string", label="选择日期")
    in_num = gr.Number(label="输入号码进行匹配：", value="")
    in_text = gr.Textbox(label="输入对话内容关键字进行搜索：", value="", info="", max_lines=100, interactive=True)
    return fe_all, in_cal, in_num, in_text, _get_cancel_btn(IS_FILTERED_A)

def btn_fe_show_listner(file: str):
    if IS_FILTERED_A:
        # 预处理文件路径
        file_name = str(file).split('/')[-1]
        file = _get_file_pth(file_name, CUR_DATE_PTH_A, AUDIO_PTH)
        if not osp.exists(file):
            raise gr.Error(f"No such file path for {file}!")
    res = select_audio_from_db(file)

    in_num = res[4]
    out_num = res[5]

    in_summary = gr.Textbox(value=res[2], label=f"呼入号码（{in_num}）今日总结：", info="", max_lines=100, interactive=True)
    out_summary = gr.Textbox(value=res[3], label=f"呼出号码（{out_num}）今日总结：", info="", max_lines=100, interactive=True)

    return _get_audio_output_component(file), res[0], res[1], in_summary, out_summary




# --- 2. 会议 --- #

def btn_fe_show_m_listner(file: str):
    if IS_FILTERED_M:
        # 预处理文件路径
        file_name = str(file).split('/')[-1]
        file = _get_file_pth(file_name, CUR_DATE_FILTERED_M, ROOT_MEETING)
        if not osp.exists(file):
            raise gr.Error(f"No such file path for {file}!")
    res = select_video_from_db(file)

    file_date, file_name = _get_date_name(file)
    file_name = "".join(file_name.split('-')[1:])
    if '.' in file_name:
        file_name = "".join(file_name.split('.')[:-1])
    video_info = gr.Markdown(f"* **会议日期：** {file_date}\n* **会议名称：** {file_name}")

    return _get_video_output_component(file), video_info, res[0], res[1],



# --- 3. 生成摘要 --- #

def process_video(pth: str):
    new_pth = pth.split('.')[0] + '.wav'
    os.system(f'ffmpeg -i {pth} -acodec pcm_s16le -f s16le -ac 1 -ar 16000 -f wav {new_pth}')
    return transcribe(new_pth)


def transcribe(inputs: str, task="transcribe"):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]

    abs = llm.invoke(f"下面是一段两个人的对话，总结一下：{text}") 

    return  text, abs


def _get_file_pth(file: str, date: str, root=AUDIO_PTH):
    return osp.join(osp.join(root, date), file)


def _get_date_name(file: str): # file: full path
    file_date = str(file).split('/')[-2]
    file_name = str(file).split('/')[-1]
    return file_date, file_name


def _get_audio_output_component(file: str):
    file_date, file_name = _get_date_name(file)
    return gr.Audio(file, label=f"{file_date} > {file_name}")

def _get_video_output_component(file: str):
    # file_date, file_name = _get_date_name(file)
    return gr.Video(file, label=None, show_label=False)


def select_audio_from_db(file: str): 
    file_date, file_name = _get_date_name(file)

    table_name='t_'+file_date
    condition=f'audio_name="{file_name}"'
    query = f'SELECT asr_result, summary, input_num_summary, output_num_summary, input_number, output_number  FROM {table_name} WHERE {condition}' # TODO

    results = _query_db(db_pth=DB_PTH_A, sql=query)

    if results:
        res = results[0]
        assert isinstance(res, tuple) and len(res)==6
        return  res[0], res[1], res[2], res[3], res[4], res[5]
    else:
        raise gr.Error(f"数据库中查询不到该文件：{file_name}！ ")


def select_video_from_db(file: str): 
    file_date, file_name = _get_date_name(file)

    table_name='t_'+file_date
    condition=f'meeting_name="{file_name}"'
    query = f'SELECT asr_result, summary FROM {table_name} WHERE {condition}' # TODO
    print(query)

    results = _query_db(db_pth=DB_PTH_M, sql=query)

    if results:
        res = results[0]
        assert isinstance(res, tuple) and len(res)==2
        return  res[0], res[1]
    else:
        print(table_name)
        print(condition)
        raise gr.Error(f"数据库中查询不到该文件：{file_name}！ ")


def _query_db(db_pth=DB_PTH_A, sql=""):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_pth)
    cursor = conn.cursor()
    cursor.execute(sql)

    # 获取查询结果
    results = cursor.fetchall() # [(, ), ...]

    # 打印结果
    # for row in results:
    #     print(len(row), '==========', row)
        # print(row[0])

    # res = results[0]
    # 关闭游标和连接
    cursor.close()
    conn.close()

    return results


def _filter_audio_files_in_db(in_date, in_num, in_text):
    date = in_date
    in_date = str(in_date).replace('-', '')
    global CUR_DATE_PTH_A

    if in_date in AUDIO_DATE_LIST:
        table_name = 't_'+in_date
        if in_num and in_text:
            print('1 ======')
            query = f"""SELECT audio_name 
                FROM {table_name} 
                WHERE (input_number = {in_num} OR output_number = {in_num}) 
                    AND asr_result LIKE '%{in_text}%'
            """

        elif in_num and not in_text:
            print('2 ======')
            query = f"""SELECT audio_name 
                FROM {table_name} 
                WHERE (input_number = {in_num} OR output_number = {in_num}) 
            """
        elif not in_num and in_text:
            print('3 ======')
            query = f"""
                SELECT audio_name 
                FROM {table_name} 
                WHERE asr_result LIKE '%{in_text}%'
            """
        else:
            print('4 ======')
            query = f"SELECT audio_name FROM {table_name}"
        
        print(query)
        results = _query_db(sql=query) # [(file_name, ),....]


        file_list = [osp.join(osp.join(AUDIO_PTH, in_date), res[0]) for res in results]


        for _file in os.listdir(TMP_PTH_A):
            os.remove(osp.join(TMP_PTH_A, _file))


        for file in file_list:
            print(file)
            shutil.copy(file, TMP_PTH_A)

        

        file_explorer_filtered = gr.FileExplorer(
            # glob="*", 
            file_count="single",
            # value=["20240408/hkhdka.txt"],
            root_dir=TMP_PTH_A,
            # ignore_glob=".**",
            # height=1000,
            label=f"日期：{date}",
            interactive=True
        )
        # gr.update(elem_id='fe_filter')

        CUR_DATE_PTH_A = in_date
        
        return file_explorer_filtered
        # return f"{in_date}, {in_num}, {in_text}"
    else:
        raise gr.Error(f"查询无该日期：{date}！ 请检查是否有名称为“{in_date}”的文件夹！")
        return "test"


def _filter_meeting_files_in_db(date: int, keyword: str) -> gr.Blocks:
    # 根据筛选条件筛选出符合的文件，输出一个gr.file_explorer组件
    # 1. 只在date所示月份下筛选
    # 2. 如不提供keyword则返回该月所有文件
    # date: 月份 (202404)
    # keyword: 摘要或者对话中的关键字

    global IS_FILTERED_M
    IS_FILTERED_M = True

    global CUR_DATE_FILTERED_M
    CUR_DATE_FILTERED_M = date

    table_name = 't_'+ date
    if keyword:
        query = f"SELECT meeting_name FROM {table_name} WHERE asr_result LIKE '%{keyword}%' "
    else:
        query = f"SELECT meeting_name FROM {table_name}"

    results = _query_db(db_pth=DB_PTH_M, sql=query) # -> [(file_name, ), ...]

    file_list = [osp.join(osp.join(ROOT_MEETING, date), res[0]) for res in results]

    for _file in os.listdir(TMP_PTH_M):
        os.remove(osp.join(TMP_PTH_M, _file))

    for file in file_list:
        print(file)
        shutil.copy(file, TMP_PTH_M)


    return gr.FileExplorer(
        file_count="single",
        root_dir=TMP_PTH_M,
        label=f"日期：{date}",
        interactive=True
    )


demo = gr.Blocks()
with demo:

    gr.Markdown(
        """
        # 🎤📜 对话文本摘要系统
        """
    )

    with gr.Tab("🎥会议"):

        gr.Markdown(
            """
            ## 使用方法
            1. 选择或输入相应筛选条件
            2. 点击“筛选”按钮对所有文件进行筛选
            3. 点击“查看”按钮查看详情
            """
        )
        with gr.Row():
            with gr.Column():
                # gr.Markdown("**筛选**：")
                with gr.Row():
                    in_year = gr.Number(
                        label="输入年份：", 
                        value=datetime.datetime.now().year, 
                        interactive=True, 
                        precision=0)

                    in_month = gr.Dropdown(
                        label="选择月份：", 
                        choices=all_months_zh,
                        value=get_cur_month_zh(),
                        allow_custom_value=False,
                        interactive=True)

                in_keyword_m = gr.Textbox(label="输入对话内容关键字进行搜索：", info="", interactive=True)
                with gr.Row():
                    btn_filter_cancel_m = gr.Button("清除", interactive=IS_FILTERED_A)
                    btn_filter_m = gr.Button("筛选", variant='primary')

                fe_all_m = gr.FileExplorer(
                    glob="*.*", 
                    file_count="single",
                    root_dir=ROOT_MEETING,
                    label="所有文件",
                )
                with gr.Row():
                    btn_all_file_m = gr.Button("📂 所有文件")
                    btn_fe_show_m = gr.Button("🔍 查看")

            
            # with gr.Row():
            with gr.Column():
                selected_video = gr.Video(value=None, label="当前会议（未选择）")
                video_info = gr.Markdown("""
                            * **会议日期：**
                            * **文件名称：**
                            """)
                meeting_summary = gr.Textbox(label="会议摘要：", info="", max_lines=35, interactive=True)
            with gr.Column():
                meeting_asr = gr.Textbox(label="会议内容", info="", max_lines=35, interactive=True)


        def tmp_update_fe_m():
            return gr.FileExplorer(root_dir=ROOT_MEETING)

        btn_filter_m.click(tmp_update_fe_m, outputs=fe_all_m).then(
            fn=btn_filter_m_listener,
            inputs=[in_year, in_month, in_keyword_m],
            outputs=[fe_all_m, btn_filter_cancel_m],
        )

        btn_filter_cancel_m.click(
            lambda _: gr.Textbox(""),
            outputs=[in_keyword_m],
        )

        btn_all_file_m.click(tmp_update_fe_m, outputs=fe_all_m).then(
            btn_all_file_m_listener,
            outputs=[fe_all_m],
        )

        btn_fe_show_m.click(
            btn_fe_show_m_listner,
            inputs=fe_all_m,
            outputs=[selected_video, video_info, meeting_asr, meeting_summary],
        )


    with gr.Tab("🎙️音频"):

        gr.Markdown(
            """
            ## 使用方法
            1. 选择或输入相应筛选条件
            2. 点击“筛选”按钮对所有文件进行筛选
            3. 点击“查看”按钮查看详情
            """
        )

        with gr.Row():
            with gr.Column():
                in_cal = Calendar(
                    type="string", 
                    label="📆 选择日期", 
                    # info="Click the calendar icon to bring up the calendar."
                )

                in_num = gr.Number(label="输入号码进行匹配：", value="")
                in_text = gr.Textbox(label="输入对话内容关键字进行搜索：", info="", max_lines=100, interactive=True)
                with gr.Row():
                    btn_filter_cancel = gr.Button("清除", interactive=IS_FILTERED_A)
                    btn_filter_a = gr.Button("筛选", variant='primary')
            
                # with gr.Column():
                fe_all = gr.FileExplorer(
                    glob="*.*", 
                    file_count="single",
                    root_dir=AUDIO_PTH,
                    # ignore_glob=".*",
                    label="所有文件",
                )
                with gr.Row():
                    btn_all_file = gr.Button("📂 所有文件")
                    btn_fe_show = gr.Button("🔍 查看")
            

                    # with gr.Column():
                    #     fe_filtered = gr.FileExplorer(
                    #         # glob="*", 
                    #         file_count="single",
                    #         # value=["20240408/hkhdka.txt"],
                    #         root_dir=TMP_PTH,
                    #         # ignore_glob=".**",
                    #         label="筛选后文件",
                    #         # elem_id='fe_filter'
                    #     )
                    #     btn_fe_filtered = gr.Button("确认")


            
            # with gr.Row():
            with gr.Column():
                selected_audio = gr.Audio(value=None, label="当前音频（未选择）")
                text_conv = gr.Textbox(label="对话", info="", max_lines=100, interactive=True)
            with gr.Column():
                text_summary = gr.Textbox(label="当前对话摘要：", info="", max_lines=100, interactive=True)
                in_summary = gr.Textbox(label="呼入号码今日总结：", info="", max_lines=100, interactive=True)
                out_summary = gr.Textbox(label="呼出号码今日总结：", info="", max_lines=100, interactive=True)


        def tmp_update_fe():
            return gr.FileExplorer(root_dir=AUDIO_PTH)

        btn_filter_a.click(tmp_update_fe, outputs=fe_all).then(
            fn=btn_filter_a_listener,
            inputs=[in_cal, in_num, in_text],
            outputs=[fe_all, btn_filter_cancel],
        )

        btn_all_file.click(tmp_update_fe, outputs=fe_all).then(
            btn_all_file_listener,
            outputs=[fe_all]
        )

        btn_filter_cancel.click(tmp_update_fe, outputs=fe_all).then(
            btn_filter_cancel_listener,
            outputs=[fe_all, in_cal, in_num, in_text, btn_filter_cancel]
        )


        btn_fe_show.click(
            btn_fe_show_listner,
            inputs=fe_all,
            outputs=[selected_audio, text_conv, text_summary, in_summary, out_summary],
        )







    with gr.Tab("🔴生成摘要"):
        gr.Markdown(
            """
            ## 使用方法
            1. 选择任意方式输入：
                - 录音
                - 上传音频
                - 上传视频
            2. 然后点击相应识别按钮
            3. 等待系统识别出文本
            4. 点击“提取摘要”按钮
            """
        )


        with gr.Row():
            with gr.Column():
                # gr.Markdown("## 录音：")
                input_mf = gr.Microphone(label='点击录音', type="filepath", show_download_button=True)
                btn_mp = gr.Button("识别录音")

                input_audio = gr.Audio(sources='upload', type="filepath", label='点击上传', )
                btn_audio = gr.Button("识别音频")

                input_video = gr.Video(sources='upload', label='点击上传')
                btn_video = gr.Button("识别视频")

            with gr.Column():
                # gr.Markdown("## 文本：")
                out_asr = gr.Textbox(label="文本", info="识别完成后可手动修改，并再次提取摘要", max_lines=100, interactive=True)
                btn_asr = gr.Button("提取摘要", visible=True)

            with gr.Column():
                # gr.Markdown("## 摘要：")
                out_abs = gr.Textbox(label="摘要", info="", max_lines=100, interactive=True)
                # btn_abs = gr.Button("保存数据库")


        btn_mp.click(
            fn=transcribe,
            inputs=input_mf,
            outputs=[out_asr, out_abs],
        )

        btn_audio.click(
            fn=transcribe,
            inputs=input_audio,
            outputs=[out_asr, out_abs],
        )

        btn_video.click(
            fn=process_video,
            inputs=input_video,
            outputs=[out_asr, out_abs],
        )






if __name__ == "__main__":
    demo.queue().launch()
    # demo.launch()
