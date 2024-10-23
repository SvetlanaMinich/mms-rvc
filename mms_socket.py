import asyncio
import json
import time
import numpy as np

import io
import soundfile as sf
import librosa    

import GPUtil

from mms_tts import MmsModels

from socketify import App, AppListenOptions, CompressOptions, WebSocket

PORT = 8081 
HOST = '0.0.0.0'

ws_count = 0

ws_queues = {}
ws_results = {}
msg_count = {}

websocket_instances = {}


async def text_to_speech(tts: MmsModels,
                         text: str,
                         language: str = 'eng'):
    func_name = f'tts_{language}'
    tts_func = getattr(tts, func_name, None)
    
    if not tts_func:
        try:
            func_name = tts.load_model(lang=language)
            tts_func = getattr(tts, func_name, None)
        except Exception:
            print(f'Cannot download model for {language} language')
            return
    
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f'        > GPU ID: {gpu.id}, Load: {gpu.load * 100}%')

    result_array, sample_rate = tts_func(text=text, device="cuda:0") 

    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f'        > GPU ID: {gpu.id}, Load: {gpu.load * 100}%')

    result_array = result_array.astype(np.float32) / np.max(np.abs(result_array))

    resampled_array = librosa.resample(result_array, orig_sr=sample_rate, target_sr=16000)

    sample_rate = 16000
    result_scaled = np.int16(resampled_array * 32767)
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, result_scaled, 16000, format='WAV')
    wav_bytes = wav_buffer.getvalue()

    num_samples = len(result_scaled)
    duration_seconds = (num_samples / sample_rate) * 1000
    print('     > duration:', duration_seconds)

    return wav_bytes 


async def process_tts_requests(tts: MmsModels,
                               ws: WebSocket,
                               client_id: str):
    queue = ws_queues[client_id]
    result_queue = ws_results[client_id]
    while True:
        # Wait for a request from the queue
        text, language, msg_id = await queue.get()

        start = time.time()
        res_in_bytes = await text_to_speech(tts=tts,
                                            text=text,
                                            language=language)
        print(f'    > {time.time() - start}: {text}')

        res_in_bytes = res_in_bytes[44:]
        
        await result_queue.put((msg_id, res_in_bytes))
        queue.task_done() 


async def send_results(ws: WebSocket, client_id: str):
    result_queue = ws_results[client_id]
    expected_id = 0
    buf = {}

    while True:
        msg_id, res_in_bytes = await result_queue.get()
        print(f'RECEIVED NUMBER {msg_id}')

        if msg_id == expected_id:
            if ws.ws not in websocket_instances:
                break
            ws.send(res_in_bytes)
            print(f"Result {len(res_in_bytes)} sent to client {client_id}, msg_id: {msg_id}")
            expected_id += 1
        if expected_id in buf.keys():
            if ws.ws not in websocket_instances:
                break
            ws.send(buf[expected_id])
            print(f"Result {len(buf[expected_id])} sent to client {client_id}, msg_id: {msg_id}")
            expected_id += 1
        if msg_id != expected_id and msg_id not in buf.keys():
            buf[msg_id] = res_in_bytes

        result_queue.task_done()
        
            

async def handle_tts(ws:WebSocket, 
                     msg, 
                     tts:MmsModels):               
    try:
        header = json.loads(msg.decode('utf-8'))
        client_id = header['Client_id']
        print(f'CLIENT {client_id} HERE\nData from client {client_id} received: {header["Text"]}, {header["Language"]}')

        if client_id not in ws_queues.keys():
            websocket_instances[ws.ws] = client_id
            ws_queues[client_id] = asyncio.Queue()
            ws_results[client_id] = asyncio.Queue()
            msg_count[client_id] = 0

            asyncio.create_task(process_tts_requests(tts=tts, ws=ws, client_id=client_id))
            asyncio.create_task(send_results(ws=ws, client_id=client_id))

        msg_id = msg_count[client_id]
        await ws_queues[client_id].put((header['Text'], header['Language'], msg_id))
        msg_count[client_id] += 1

    except Exception as ex:
        print(f'     > Error occured: {ex}')


def on_open(ws: WebSocket):
    websocket_instances[ws.ws] = ''
    print(f"WebSocket {ws.ws} opened")


def on_close(ws: WebSocket, code, msg):
    if ws.ws in websocket_instances.keys():
        client_id = websocket_instances.pop(ws.ws)
        ws_results.pop(client_id)
        msg_count.pop(client_id)
        ws_queues.pop(client_id)
        print(f"Client {client_id} removed")
        print(f"WebSocket {ws.ws} closed")
    else:
        print('WebSocket closed, but not in storage')


def run_server_tts():
    tts = MmsModels()
    
    app = App()
    app.ws('/', {
                "compression": CompressOptions.SHARED_COMPRESSOR,
                "max_payload_length": 16 * 1024 * 1024,
                "idle_timeout": 960, 
                "open": on_open,
                "message": lambda ws, msg, opcode: asyncio.create_task(
                        handle_tts(ws, msg, tts)),
                "close": on_close
            })
    app.listen(AppListenOptions(PORT, HOST), lambda config: print(f"Listening on port {config.port}"))
    app.run()


if __name__ == "__main__":
    try:
        run_server_tts()
    except Exception as ex:
        print(ex)

