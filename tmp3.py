from progress.bar import ChargingBar
from progress.spinner import Spinner
import time

spinner = Spinner()
bar = ChargingBar()
bar.max = 200
bar.width = 0

for i in range(20):
    time.sleep(1)
    bar.next()
    spinner.next()