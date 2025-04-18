import numpy as np
import re
import re
import numpy as np

import re
import numpy as np

import re
import numpy as np

def extract_points(generate_param, image_w, image_h):
    all_points = []
    pattern = r'x\d*=\s*\\?"?([0-9]+(?:\.[0-9]*)?)\\?"?\s*y\d*=\s*\\?"?([0-9]+(?:\.[0-9]*)?)\\?"?'
    
    for match in re.finditer(pattern, generate_param):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            continue
        else:
            point = np.array(point)
            if np.max(point) > 100:
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    
    return all_points
test_cases = [
    'x1="49.5" y1="94.0" x2="70.0" y2="94.2"',
    'x1="10.0" y1="20.0" x2="30.5" y2="40.5"',
    'x3="150.0" y3="200.0" x4="99.9" y4="100.1"',
    'x1="1.0" y1="2.0" x2="3.0" y2="4.0"',
    'x5="0.0" y5="0.0" x6="50.5" y6="60.5"',
    'x7="102.0" y7="110.5" x8="300.0" y8="400.0"',
    'x="49.5" y="94.0" x_2="70.0" y_2="94.2"',
    'x_1="10.0" y_1="20.0" x_3="30.5" y_3="40.5"',
    'x1=\"49.5\" y1=\"94.0\" x2=\"70.0\" y2=\"94.2\"',
    "x=49.5 y=94.0",
]

image_w = 1920
image_h = 1080

for idx, generate_param in enumerate(test_cases, 1):
    print(f"Test {idx} - Input: {generate_param}")
    points = extract_points(generate_param, image_w, image_h)
    for point in points:
        print(f"  Point: {point}")
    print('-' * 30)
