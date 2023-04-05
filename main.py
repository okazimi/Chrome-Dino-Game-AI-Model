# # # AI MODEL
# # IMPORTS
# WEB GAME ENVIRONMENT IMPORT
from environment import WebGame
# AI MODEL TRAIN AND LOGGING IMPORT
from trainandcallback import TrainAndLoggingCallback
# DEEP Q-NETWORK IMPORT (https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
from stable_baselines3 import DQN
# VISUALIZE CAPTURED FRAMES IMPORT
from matplotlib import pyplot as plt
# TIME IMPORT (PAUSES)
import time
# ARRAY IMPORT
import numpy as np

# INITIALIZE DIRECTORIES
CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"

# CODE TO RUN WHEN EXECUTED AS A SCRIPT
# COOL EXPLANATION OF IF __NAME__ == "__MAIN__": https://realpython.com/if-name-main-python/
if __name__ == "__main__":
    # INITIALIZE ENVIRONMENT
    env = WebGame()
    # INITIALIZE TRAINING AND LOGGING OF AI MODEL
    callback = TrainAndLoggingCallback(check_freq=50000,  # CHECK_FREQ: SAVE AN INSTANCE OF OUR MODEL EVERY 50000 FRAMES
                                       save_path=CHECKPOINT_DIR) # WHERE TO SAVE EACH MODEL INSTANCE
    # INITIALIZE DQN AI MODEL (DEEP Q-NETWORK)
    model = DQN("CnnPolicy",  # CNNPOLICY: WHEN USING IMAGES AN INPUT
                env,  # THE ENVIRONMENT TO LEARN FROM
                tensorboard_log=LOG_DIR,  # THE LOG LOCATION FOR TENSORBOARD
                verbose=1,  # VERBOSITY LEVEL: 0 FOR NO OUTPUT, 1 FOR INFO MESSAGES, 2 FOR DEBUG MESSAGES
                buffer_size=40000,  # SIZE OF THE REPLAY BUFFER
                learning_starts=0)  # HOW MANY STEPS OF THE MODEL TO COLLECT BEFORE LEARNING STARTS
    # BEGIN TRAINING :)
    model.learn(total_timesteps=69000,  # TRAIN MODEL FOR 88000 STEPS (FRAMES)
                callback=callback)  # ENSURE WE SAVE OUR MODEL

    # # # TEST PREVIOUSLY SAVED MODEL
    # # LOAD MODEL
    # # model.load(FULL PATH HERE)
    # # FOR 10 EPISODES
    # for episode in range(10):
    #     # RESET GAME AND GET OBSERVATION (SCREEN CAPTURE OF GAME)
    #     obs = env.reset()
    #     # SET GAME_OVER TO FALSE
    #     game_over = False
    #     # SET TOTAL_REWARD TO 0
    #     total_reward = 0
    #     # WHILE GAME IS NOT OVER
    #     while not game_over:
    #         # PASS OBSERVATION TO AI MODEL AND HAVE IT PREDICT ACTION
    #         action, _ = model.predict(obs)
    #         # PASS PREDICTED ACTION TO STEP FUNCTION
    #         # RETURN OBSERVATION, REWARD, DONE STATUS AND INFO
    #         obs, reward, game_over, info = env.step(int(action))
    #         # INCREMENT TOTAL REWARDS
    #         total_reward += reward
    #         # PRINT STATEMENT
    #     print(f"Total rewards for episode {episode} is {total_reward}")


# # # NON AI MODEL
# # SELENIUM IMPORTS
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# # IMAGE PROCESSING IMPORTS
# from PIL import ImageGrab
# import pyautogui
# # TIME IMPORT
# import time
#
#
# # SELENIUM: OPEN BROWSER TAB, MAXIMIZE WINDOW AND NAVIGATE TO TARGET URL
# def web_request_automation(target_url):
#     # INITIALIZE CHROME WEB-DRIVER OPTIONS OBJECT
#     chrome_options = Options()
#     # SET OPTION "DETACH" == TRUE (KEEP BROWSER OPEN)
#     chrome_options.add_experimental_option("detach", True)
#     # INITIALIZE CHROME WEB-DRIVER
#     chrome = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
#     # MAXIMIZE CHROME WINDOW
#     chrome.maximize_window()
#     # NAVIGATE TO TARGET URL
#     chrome.get(target_url)
#
#
# # PYAUTOGUI: PRESS KEYBOARD KEY
# def press(key):
#     # PRESS KEYBOARD KEY
#     pyautogui.press(key)
#
#
# # CACTUS COLLISION CHECKER
# def check_cactus_collision(pixel_data):
#     # BLOCK START AND END X-POSITION
#     for i in range(995, 1325):
#         # BLOCK START AND END Y-POSITION
#         for j in range(1010, 1060):
#             # CHECK FOR CACTUS BASED ON COLOR (100 GRAYSCALE)
#             if pixel_data[i, j] < 100:
#                 # PRESS UP KEY
#                 press("up")
#                 # RETURN TO IMAGE PROCESSING (WHILE LOOP)
#                 return
#
#
# # BIRD COLLISION CHECKER
# def check_bird_collision(pixel_data):
#     # BLOCK START AND END X-POSITION
#     for i in range(950, 1150):
#         # BLOCK START AND END Y-POSITION
#         for j in range(959, 1009):
#             # CHECK FOR BIRD BASED ON COLOR (171 GRAYSCALE)
#             if pixel_data[i, j] < 171:
#                 # PRESS DOWN KEY
#                 press('down')
#                 # RETURN TO IMAGE PROCESSING (WHILE LOOP)
#                 return
#
#
# # GAME OVER CHECKER
# def game_over_checker(pixel_data):
#     # BLOCK START AND END X-POSITION
#     for i in range(1100, 1600):
#         # BLOCK START AND END Y-POSITION
#         for j in range(825, 900):
#             # CHECK FOR GAME OVER SIGN BASED ON COLOR (171 GRAYSCALE)
#             if pixel_data[i, j] < 171:
#                 # PRESS UP KEY
#                 press('up')
#                 # RETURN TO IMAGE PROCESSING (WHILE LOOP)
#                 return
#
#
# # CODE TO RUN WHEN EXECUTED AS A SCRIPT
# # COOL EXPLANATION OF IF __NAME__ == "__MAIN__": https://realpython.com/if-name-main-python/
# if __name__ == "__main__":
#     # SELENIUM: AUTOMATED GET REQUEST TO TARGET URL
#     web_request_automation("https://trex-runner.com/")
#     # WAIT FOR WEB PAGE TO RENDER
#     time.sleep(2)
#     # PYAUTOGUI: PRESS UP KEY TO BEGIN GAME
#     press('up')
#     # WAIT FOR GAME VISUALS TO RENDER
#     time.sleep(1)
#     # WHILE LOOP TO CONSTANTLY MONITOR SCREEN
#     while True:
#         # CAPTURE CURRENT SCREEN AND CONVERT TO GRAYSCALE (BETTER IMAGE PROCESSING)
#         image = ImageGrab.grab().convert("L")
#         # OBTAIN PIXEL DATA FROM THE CURRENT CAPTURED SCREEN
#         pixel_data = image.load()
#         # CHECK CACTUS COLLISION
#         check_cactus_collision(pixel_data)
#         # CHECK BIRD COLLISION
#         check_bird_collision(pixel_data)
#         # CHECK GAME OVER
#         game_over_checker(pixel_data)
