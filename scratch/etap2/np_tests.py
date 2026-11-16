import numpy as np
from datetime import datetime

now = datetime.now()

formatted = now.strftime("%Y-%m-%d_%H:%M:%S")

print(formatted)