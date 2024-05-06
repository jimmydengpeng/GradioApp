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


TEST_LOAD_MODEL = False
if TEST_LOAD_MODEL:
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=device)
    llm = Ollama(model="qwen:7b")





# --- Button Click Listeners ---

def btn_fe_all_listner(file: str):
    res = get_file_asr_summary_in_db(file)
    return _get_audio_output_component(file), res[0], res[1]


# 将tmp目录里的file与audio目录对应的匹配
def btn_fe_filtered_listener(file_tmp: str):
    # 预处理文件路径
    file_name = str(file_tmp).split('/')[-1]
    file = _get_file_pth(file_name, CUR_DATE_PTH, AUDIO_PTH)

    if not osp.exists(file):
        raise gr.Error(f"No such file path for {file}!")
    res = get_file_asr_summary_in_db(file)
    return _get_audio_output_component(file), res[0], res[1]


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


def _get_audio_output_component(pth: str):
    return gr.Audio(pth, label="当前音频")
 

def get_file_asr_summary_in_db(file: str): # file: full path
    file_date = str(file).split('/')[-2]
    file_name = str(file).split('/')[-1]

    table_name='t_'+file_date
    condition=f'audio_name="{file_name}"'
    query = f'SELECT asr_result, summary FROM {table_name} WHERE {condition}' # TODO

    results = _query_db(db_pth=DB_PTH, sql=query)

    if results:
        res = results[0]
        assert isinstance(res, tuple) and len(res)==2
        return res[0], res[1]
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


def filter_files_in_db(in_date, in_num, in_text):
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
            label="筛选后文件",
            interactive=True
            
            # elem_id='fe_123',
        )
        # gr.update(elem_id='fe_filter')

        CUR_DATE_PTH = in_date
        
        return file_explorer_filtered
        # return f"{in_date}, {in_num}, {in_text}"
    else:
        raise gr.Error(f"查询无该日期：{date}！ 请确保文件夹名称的格式如：20240102！")
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
                fe_all = gr.FileExplorer(
                    glob="*", 
                    file_count="single",
                    # value=["20240408/hkhdka.txt"],
                    root_dir=AUDIO_PTH,
                    # ignore_glob=".**",
                    label="所有文件",
                )
                btn_fe_all = gr.Button("确认")
            

            with gr.Column():
                in_cal = Calendar(
                    type="string", 
                    label="选择日期", 
                    # info="Click the calendar icon to bring up the calendar."
                )
                # btn_cal = gr.Button("确认")

                in_num = gr.Number(label="输入号码进行匹配：", value="")
                # btn_num = gr.Button("确认")

                in_text = gr.Textbox(label="输入关键字进行搜索：", info="", max_lines=100, interactive=True)
                btn_filter = gr.Button("筛选")

                # text_test = gr.Textbox(label="", info="", max_lines=100, interactive=True)


                with gr.Column():
                    fe_filtered = gr.FileExplorer(
                        # glob="*", 
                        file_count="single",
                        # value=["20240408/hkhdka.txt"],
                        root_dir=TMP_PTH,
                        # ignore_glob=".**",
                        label="筛选后文件",
                        # elem_id='fe_filter'
                    )
                    btn_fe_filtered = gr.Button("确认")


            
            # with gr.Column():
            # with gr.Row():
            text_conv = gr.Textbox(label="对话", info="", max_lines=100, interactive=True)
            with gr.Column():
                text_summary = gr.Textbox(label="总结", info="", max_lines=100, interactive=True)
                selected_audio = gr.Audio(value=None, label="当前音频（未选择）")




        btn_fe_all.click(
            btn_fe_all_listner,
            inputs=fe_all,
            outputs=[selected_audio, text_conv, text_summary],
        )

        def tmp_update_fe():
            return gr.FileExplorer(root_dir=AUDIO_PTH)

        btn_filter.click(tmp_update_fe, outputs=fe_filtered).then(
            fn=filter_files_in_db,
            inputs=[in_cal, in_num, in_text],
            outputs=[fe_filtered],
        )

        btn_fe_filtered.click(
            btn_fe_filtered_listener,
            inputs=fe_filtered,
            outputs=[selected_audio, text_conv, text_summary],
        )



        # btn_filter.click(fn=lambda value='./tmp/': gr.update(root_dir=value), inputs=[in_text], outputs=file_filtered)
  #    gr.Button("确认")


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







    # with gr.Tab("📁上传文件"):
    #     gr.Markdown(
    #         """
    #         ## 使用方法
    #         1. 先点击上传按钮上传音频文件
    #         2. 然后等待系统处理
    #         """
    #     )
    #     gr.Audio(sources='upload', label='点击上传', )
    #     gr.Button("确认")








if __name__ == "__main__":
    demo.queue().launch()
    # demo.launch()

