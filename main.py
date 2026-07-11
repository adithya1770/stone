import time
from telemetry import Telemetry
from model_manager import ModelManager
from decision_engine import DecisionEngine
from logger import Logger

ACTION_TO_MODEL_TYPE = {0: "fp32", 1: "int8"}

telemetry = Telemetry()
mm = ModelManager("models/mobilenet_v2_fp32.tflite", "models/mobilenet_v2_int8.tflite", "models/labels.txt")
engine = DecisionEngine(min_mode_duration=5)
logger = Logger("adaptive_run.csv")

input_id = 0
try:
    while True:
        before_snap = telemetry.read()
        action = engine.decide(before_snap)
        model_type = ACTION_TO_MODEL_TYPE[action]

        result = mm.predict("dog.jpeg", model_type)

        after_snap = telemetry.read()

        logger.log(
            before_snap=before_snap,
            after_snap=after_snap,
            input_id=input_id,
            chosen_action=action,
            model_used=model_type,
            inference_result=result,
        )

        print(before_snap, result)
        input_id += 1
        time.sleep(3)
except KeyboardInterrupt:
    logger.close()
    print("Stopped, log saved.")