# # IMPORTS
# OPERATING SYSTEM IMPORT
import os
# SCREEN CAPTURE IMPORT
from mss import mss
# KEYBOARD CONTROL IMPORT
import pydirectinput
# EXTRACT GAME OVER SIGN IMPORT (OCR)
import pytesseract
# TIME IMPORT (PAUSES)
import time
# ENVIRONMENT IMPORTS
from gym import Env
from gym.spaces import Box, Discrete
# FRAME PROCESSING IMPORT
import cv2
# ARRAY IMPORT
import numpy as np


# # BUILD ENVIRONMENT
# WEB GAME = SUBCLASS (CHILD)
# ENV = SUPERCLASS (PARENT)
class WebGame(Env):

    # INITIALIZE ENVIRONMENT, ACTION AND OBSERVATION SHAPES
    def __init__(self):
        # INHERITANCE: ACCESS ENV'S INIT METHOD AND METHODS
        super().__init__()
        # INITIALIZE OBSERVATION SPACE
        # LOW: 0 (LOWER BOUND OF INTERVALS IN ARRAY) (PRESUMABLY COLOR)
        # HIGH: 255 (UPPER BOUND OF INTERVALS IN ARRAY) (PRESUMABLY COLOR)
        # SHAPE: 1 BATCH, 83 PIXEL HIGH, 100 PIXELS WIDE (IMAGE SIZE)
        # DTYPE: NP.UINT8 (SMALL AND COMPRESSED DATA TYPE)
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        # INITIALIZE ACTION SPACE
        # NUMBER OF POSSIBLE ACTIONS: 3 (0 (UP), 1 (DOWN), 2 (NO OPERATION))
        self.action_space = Discrete(3)
        # INITIALIZE MSS CLASS TO ENABLE SCREEN CAPTURE
        self.cap = mss()
        # REGION OF THE GAME SCREEN TO CAPTURE (DINO & CACTUS & BIRDS)
        # 300 PIXELS FROM THE TOP, BEGIN FROM THE LEFT, 600 PIXELS WIDE AND 500 PIXELS HIGH ######## EDIT LATER #########################################
        self.game_location = {'top': 850, 'left': 0, 'width': 850, 'height': 500}
        # REGION OF THE GAME SCREEN TO CAPTURE (GAME OVER SIGN)
        # 405 PIXELS FROM THE TOP, BEGIN 630 PIXELS FROM THE LEFT, 660 PIXELS WIDE AND 70 PIXELS HIGH ######## EDIT LATER ###############################
        self.done_location = {'top': 825, 'left': 860, 'width': 975, 'height': 100}

    # GAME ACTION
    def step(self, action):
        # ACTION KEYS: 0 = UP, 1 = DOWN, 2 = NO ACTION
        # CREATE ACTION_MAP DICTIONARY
        action_map = {
            0: "space",
            1: "down",
            2: "no-op"
        }
        # IF ACTION != 2 (NO OPERATION)
        if action != 2:
            # PRESS SPACE OR DOWN KEY
            pydirectinput.press(action_map[action])
        # CHECK FOR GAME OVER SIGN
        game_over, game_over_cap = self.get_done()
        # GET NEXT OBSERVATION
        next_observation = self.get_observation()
        # SET REWARD (WE GET A POINT FOR EVERY FRAME WE'RE ALIVE)
        reward = 1
        # INFO DICTIONARY (REQUIRED FOR STABLE-BASELINE)
        info = {}
        # RETURN NEXT OBSERVATION, REWARD, GAME OVER STATUS AND INFO DICTIONARY
        return next_observation, reward, game_over, info

    # VISUALIZE GAME ENVIRONMENT
    def render(self):
        # GET SCREEN CAPTURE OF GAME
        # CONVERT SCREEN CAPTURE TO NP.ARRAY (PIXEL VALUES)
        # GRAB ALL HEIGHT ARRAYS, ALL WIDTH ARRAYS AND FIRST 3 ARRAY VALUES (500, 600, 3)
        # HAVE CV2 SHOW IMAGE
        cv2.imshow("Game", np.array(self.cap.grab(self.game_location))[:, :, :3])
        # KEEP CV2 IMAGE OPEN UNTIL USER PRESSES "Q" KEY
        if cv2.waitKey(0) & 0xFF == ord('q'):
            # CALL SELF.CLOSE METHOD (DESTROY ALL CV2 WINDOWS)
            self.close()

    # END GAME ENVIRONMENT OBSERVATION
    def close(self):
        # CLOSE ALL OPEN CV2 WINDOWS
        cv2.destroyAllWindows()

    # RESTART GAME
    def reset(self):
        # WAIT ONE SECOND TO BEGIN RESET
        time.sleep(1)
        # CLICK SOMEWHERE ON THE SCREEN TO ASSURE RESET HAPPENS
        pydirectinput.click(x=300, y=300)
        # PRESS SPACE BAR TO RESET GAME
        pydirectinput.press('space')
        # RETURN SCREEN CAPTURE OF GAME
        return self.get_observation()

    # CAPTURE DESIRED GAME SCREEN
    def get_observation(self):
        # GET SCREEN CAPTURE OF GAME
        # CONVERT SCREEN CAPTURE TO NP.ARRAY (PIXEL VALUES) WITH A D-TYPE OF UINT8 (FASTER PROCESSING)
        # GRAB ALL HEIGHT ARRAYS, ALL WIDTH ARRAYS AND FIRST 3 ARRAY VALUES (500, 600, 3)
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3].astype(np.uint8)
        # CONVERT RAW PIXEL VALUES TO GRAYSCALE
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # RESIZE: 100 PIXELS WIDE * 83 PIXELS HIGH
        resized = cv2.resize(gray, (100, 83))
        # ADD CHANNELS FIRST (RESHAPE FOR STABLE-BASELINE) (83 PIXELS HIGH * 100 PIXELS WIDE)
        channel = np.reshape(resized, (1, 83, 100))
        # RETURN CHANNEL
        return channel

    # GAME OVER CHECKER (USING OCR)
    def get_done(self):
        # GET SCREEN CAPTURE OF GAME OVER SIGN
        # CONVERT SCREEN CAPTURE TO NP.ARRAY (PIXEL VALUES) WITH A D-TYPE OF UINT8 (FASTER PROCESSING)
        # GRAB ALL HEIGHT ARRAYS, ALL WIDTH ARRAYS AND FIRST 3 ARRAY VALUES (500, 600, 3)
        game_over_cap = np.array(self.cap.grab(self.done_location))[:, :, :3]
        # VALID GAME OVER TEXT (WHAT WE'RE LOOKING FOR)
        game_over_string = ["GAME", "GAHE"]
        # INITIALIZE GAME OVER STATUS VARIABLE AND SET TO FALSE
        game_over = False
        # SPECIFY PYTESSERACT PATH
        pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_PATH")
        # OPTICAL CHARACTER RECOGNITION (OCR)
        # CONVERT GAME OVER CAPTURE TO STRING VALUE AND OBTAIN FIRST FOUR LETTERS
        res = pytesseract.image_to_string(game_over_cap)[:4]
        # IF PYTESSERACT (OCR) RESPONSE MATCHES VALID GAME OVER TEXT
        if res in game_over_string:
            # SET GAME OVER STATUS TO TRUE
            game_over = True
        # RETURN GAME OVER STATUS AND GAME OVER SCREEN CAPTURE
        return game_over, game_over_cap
