import time
import asyncio
from threading import Thread

import cv2
import numpy as np

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


class VideoServer:
    def __init__(self, config):
        self.config_server = config["web_stream"]
        self.app = FastAPI()

        # Routes
        self.app.get("/video")(self._update_page)

        self.output_size = config["show"]["output_size"]
        self.host_ip = self.config_server["host_ip"]
        self.port = self.config_server["port"]
        self.fps = self.config_server["fps"]
        self.show_delay = 1 / self.fps

        self.server_thread = None
        self._frame = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)
        self.last_update = time.time()

    async def _gen(self):
        while True:
            now = time.time()

            if now - self.last_update < self.show_delay:
                await asyncio.sleep(self.show_delay - (now - self.last_update))
            ret, jpeg = cv2.imencode(".jpg", self._frame)
            encoded_image = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded_image + b"\r\n\r\n"
            )
            self.last_update = time.time()

    def _update_page(self):
        return StreamingResponse(self._gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    def update_image(self, image: cv2.typing.MatLike):
        self._frame = image

    def run(self):
        def start_server():
            uvicorn.run(self.app, host=self.host_ip, port=self.port)

        self.server_thread = Thread(target=start_server, daemon=True)
        self.server_thread.start()
