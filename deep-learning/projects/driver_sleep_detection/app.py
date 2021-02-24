from live_detection import LiveDetection
import tensorflow as tf

if __name__ == "__main__":
    driver_name = input(str("Enter Driver's Name:"))
    l_d = LiveDetection(driver_name)
    l_d.start()
    
