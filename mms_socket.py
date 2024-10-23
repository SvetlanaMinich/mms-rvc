import asyncio
import json
import time
import numpy as np
import scipy.io.wavfile as wavfile
import io

from mms_tts import MmsModels
from WebsocketLocal import WSLocal

from socketify import App, AppListenOptions, CompressOptions, WebSocket

PORT = 8081 
HOST = '0.0.0.0'

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
        
    result_ndarray, sample_rate = tts_func(text=text) 

    print('tts done in tts')
    # sample_rate = 16000  # Target sample rate (Hz)

    result_scaled = np.int16(result_ndarray * 32767)
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, result_scaled)
        
    return wav_buffer.getvalue()  


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
        
        res_in_bytes = res_in_bytes[44:]

        print(f'    > {time.time() - start}: {text}')
        
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
            ws.send(res_in_bytes)
            print(f"Result {len(res_in_bytes)} sent to client {client_id}, msg_id: {msg_id}")
            expected_id += 1
        if expected_id in buf.keys():
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
    new_ws = WSLocal()
    websocket_instances[ws] = new_ws
    print(f'WebSocket {new_ws.get_id()} opened')


def on_close(ws: WebSocket):
    if ws in websocket_instances.keys():
        ws_instance = websocket_instances.pop(ws)  # Удаляем инстанс при закрытии
        closed_ws_id = ws_instance.on_close()
        print(f'WebSocket {closed_ws_id} closed')
    else:
        print('WebSocket closed, but not in storage')


def run_server_tts():
    tts = MmsModels()
    
    app = App()
    app.ws('/', {
                "compression": CompressOptions.SHARED_COMPRESSOR,
                "max_payload_length": 16 * 1024 * 1024,
                "idle_timeout": 960, 
                "open": lambda ws: on_open(ws),
                "message": lambda ws, msg, opcode: asyncio.create_task(
                        handle_tts(ws, msg, tts)),
                "close": lambda ws, code, msg: on_close(ws)
            })
    app.listen(AppListenOptions(PORT, HOST), lambda config: print(f"Listening on port {config.port}"))
    app.run()


if __name__ == "__main__":
    try:
        run_server_tts()
    except Exception as ex:
        print(ex)

