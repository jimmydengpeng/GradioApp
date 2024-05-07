import os, shutil
import os.path as osp
import numpy as np
import gradio as gr
from gradio_calendar import Calendar
import sqlite3
import torch
from transformers import pipeline
from langchain_community.llms import Ollama


MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000
VIDEO_LENGTH_LIMIT_S = 3600  # TODO(jimmy) limit to 1 hour video files

device = 0 if torch.cuda.is_available() else "cpu"


# --- 设置全局路径 --- 
# 音频文件所在路径
AUDIO_PTH = './audio'
AUDIO_FILE_LIST = os.listdir(AUDIO_PTH)

# 数据库文件路径
DB_PTH = 'audio.db'

# 临时文件目录
TMP_PTH = './tmp/'
if not osp.exists(TMP_PTH):
    os.mkdir(TMP_PTH)

CUR_DATE_PTH = AUDIO_FILE_LIST[0] # e.g. 2024XXXX
IS_FILTERED = False


TEST_LOAD_MODEL = False
if TEST_LOAD_MODEL:
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=device)
    llm = Ollama(model="qwen:7b")


# --- Button Click Listeners ---

def btn_filter_listener(in_date, in_num, in_text):
    global IS_FILTERED
    IS_FILTERED = True
    return _filter_files_in_db(in_date, in_num, in_text), _get_cancel_btn(IS_FILTERED)

def _get_fe_all():
    return gr.FileExplorer(
        glob="*.*", 
        file_count="single",
        root_dir=AUDIO_PTH,
        # ignore_glob=".*",
        label="所有文件")

def _get_cancel_btn(interactive):
    return gr.Button("清除", interactive=interactive)

def btn_all_file_listener():
    global IS_FILTERED
    IS_FILTERED = False
    return _get_fe_all()

def btn_filter_cancel_listener():
    global IS_FILTERED
    IS_FILTERED = False
    fe_all = _get_fe_all()
    in_cal = Calendar(value=None, type="string", label="选择日期")
    in_num = gr.Number(label="输入号码进行匹配：", value="")
    in_text = gr.Textbox(label="输入对话内容关键字进行搜索：", value="", info="", max_lines=100, interactive=True)
    return fe_all, in_cal, in_num, in_text, _get_cancel_btn(IS_FILTERED)

def btn_fe_show_listner(file: str):
    if IS_FILTERED:
        # 预处理文件路径
        file_name = str(file).split('/')[-1]
        file = _get_file_pth(file_name, CUR_DATE_PTH, AUDIO_PTH)
        if not osp.exists(file):
            raise gr.Error(f"No such file path for {file}!")
    res = get_file_asr_summary_in_db(file)

    in_num = res[4]
    out_num = res[5]

    in_summary = gr.Textbox(value=res[2], label=f"呼入号码（{in_num}）今日总结：", info="", max_lines=100, interactive=True)
    out_summary = gr.Textbox(value=res[3], label=f"呼出号码（{out_num}）今日总结：", info="", max_lines=100, interactive=True)

    return _get_audio_output_component(file), res[0], res[1], in_summary, out_summary


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
 

def get_file_asr_summary_in_db(file: str): 
    file_date, file_name = _get_date_name(file)

    table_name='t_'+file_date
    condition=f'audio_name="{file_name}"'
    query = f'SELECT asr_result, summary, input_num_summary, output_num_summary, input_number, output_number  FROM {table_name} WHERE {condition}' # TODO

    results = _query_db(db_pth=DB_PTH, sql=query)

    if results:
        res = results[0]
        assert isinstance(res, tuple) and len(res)==6
        return  res[0], res[1], res[2], res[3], res[4], res[5]
    else:
        raise gr.Error(f"数据库中查询不到该文件：{file_name}！ ")


def _query_db(db_pth=DB_PTH, sql=""):
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


def _filter_files_in_db(in_date, in_num, in_text):
    date = in_date
    in_date = str(in_date).replace('-', '')
    global CUR_DATE_PTH

    if in_date in AUDIO_FILE_LIST:
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


        for _file in os.listdir(TMP_PTH):
            os.remove(osp.join(TMP_PTH, _file))


        for file in file_list:
            print(file)
            shutil.copy(file, TMP_PTH)

        

        file_explorer_filtered = gr.FileExplorer(
            # glob="*", 
            file_count="single",
            # value=["20240408/hkhdka.txt"],
            root_dir=TMP_PTH,
            # ignore_glob=".**",
            # height=1000,
            label=f"日期：{date}",
            interactive=True
        )
        # gr.update(elem_id='fe_filter')

        CUR_DATE_PTH = in_date
        
        return file_explorer_filtered
        # return f"{in_date}, {in_num}, {in_text}"
    else:
        raise gr.Error(f"查询无该日期：{date}！ 请检查是否有名称为“{in_date}”的文件夹！")
        return "test"



demo = gr.Blocks()
with demo:

    gr.Markdown(
        """
        # 🎤📜 对话文本摘要系统
        """
    )

    with gr.Tab("📁️浏览文件"):

        gr.Markdown(
            """
            ## 使用方法
            1.
            """
        )

        with gr.Row():
            with gr.Column():
                in_cal = Calendar(
                    type="string", 
                    label="选择日期", 
                    # info="Click the calendar icon to bring up the calendar."
                )

                in_num = gr.Number(label="输入号码进行匹配：", value="")
                in_text = gr.Textbox(label="输入对话内容关键字进行搜索：", info="", max_lines=100, interactive=True)
                with gr.Row():
                    btn_filter_cancel = gr.Button("清除", interactive=IS_FILTERED)
                    btn_filter = gr.Button("筛选", variant='primary')
            
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

        btn_filter.click(tmp_update_fe, outputs=fe_all).then(
            fn=btn_filter_listener,
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



    with gr.Tab("🎙️生成摘要"):
        gr.Markdown(
            """
            ## 使用方法
            1. 先点击录音
            2. 然后等待系统处理
            """
        )


        with gr.Row():
            with gr.Column():
                # gr.Markdown("## 录音：")
                input_mf = gr.Microphone(label='点击录音', type="filepath", show_download_button=True)
                btn_mp = gr.Button("识别录音")

                input_audio = gr.Audio(sources='upload', type="filepath", label='点击上传', )
                btn_audio = gr.Button("识别音频")

                input_video = gr.Video(sources='upload')
                btn_video = gr.Button("识别视频")

            with gr.Column():
                # gr.Markdown("## 文本：")
                out_asr = gr.Textbox(label="文本", info="识别完成后可手动修改，并再次提取摘要", max_lines=100, interactive=True)
                btn_asr = gr.Button("提取摘要", visible=True)

            with gr.Column():
                # gr.Markdown("## 摘要：")
                out_abs = gr.Textbox(label="摘要", info="提取完成后可手动修改摘要再保存", max_lines=100, interactive=True)
                btn_abs = gr.Button("保存数据库")


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
