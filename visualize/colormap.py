import numpy as np

def nws_precip_colors():

    nan_zero = [
            "#f0f0f0",  #  nan
            "#ffffff"   #  0.00 
    ]

    nws_precip_colors_original = [        
            "#04e9e7",  #  0.01
            "#019ff4",  #  5.00
            "#0300f4",  # 10.00
            "#02fd02",  # 15.00
            "#01c501",  # 20.00
            "#008e00",  # 25.00
            "#fdf802",  # 30.00
            "#e5bc00",  # 35.00
            "#fd9500",  # 40.00
            "#fd0000",  # 45.00
            "#d40000",  # 50.00
            "#bc0000",  # 55.00
            "#f800fd",  # 60.00
            "#9854c6",  # 65.00
        ]

# In [5]:
# nws_precip_colors = [
#     "#04e9e7",  # 0.01 - 0.10 inches
#     "#019ff4",  # 0.10 - 0.25 inches
#     "#0300f4",  # 0.25 - 0.50 inches
#     "#02fd02",  # 0.50 - 0.75 inches
#     "#01c501",  # 0.75 - 1.00 inches
#     "#008e00",  # 1.00 - 1.50 inches
#     "#fdf802",  # 1.50 - 2.00 inches
#     "#e5bc00",  # 2.00 - 2.50 inches
#     "#fd9500",  # 2.50 - 3.00 inches
#     "#fd0000",  # 3.00 - 4.00 inches
#     "#d40000",  # 4.00 - 5.00 inches
#     "#bc0000",  # 5.00 - 6.00 inches
#     "#f800fd",  # 6.00 - 8.00 inches
#     "#9854c6",  # 8.00 - 10.00 inches
#     "#fdfdfd"   # 10.00+
# ]
    color_int = []
    print("show i==1 ")
    # print(np.linspace(int(nws_precip_colors_original[1][1:3], 16), int(nws_precip_colors_original[1+1][1:3], 16), 500))
    for i, val in enumerate(nws_precip_colors_original[:-1]):
        # print("i=",i," val=",val)
        red = [val for val in np.linspace(int(nws_precip_colors_original[i][1:3], 16), int(nws_precip_colors_original[i+1][1:3], 16), 500)]
        green = [val for val in np.linspace(int(nws_precip_colors_original[i][3:5], 16), int(nws_precip_colors_original[i+1][3:5], 16), 500)]
        blue = [val for val in np.linspace(int(nws_precip_colors_original[i][5:7], 16), int(nws_precip_colors_original[i+1][5:7], 16), 500)]
        # print("blue=",np.array(blue).shape)
        # stack2 = np.vstack([red, green, blue])
        # print("stack2=",stack2.shape)
        
        stack = np.vstack([red, green, blue]).T
        # print("stack=",stack.shape)
        color_int = stack if color_int == [] else np.concatenate((color_int, stack))
        # print("color_int=",color_int.shape)

        # i= 0  val= #04e9e7
        # blue= (500,)
        # stack2= (3, 500)
        # stack= (500, 3)
        # color_int= (500, 3)
        # ----
        # i= 12  val= #f800fd
        # blue= (500,)
        # stack2= (3, 500)
        # stack= (500, 3)
        # color_int= (6500, 3)
        # print('----')
    # print("=============")
    print("color_int=",color_int.shape)
    color_code = []
    for val in color_int:
        color_code.append('#{:02X}{:02X}{:02X}'.format(int(val[0]), int(val[1]), int(val[2])))
    print("color_code=",np.array(color_code).shape)
    color_code = np.concatenate([nan_zero, color_code])
    print("color_code=",color_code.shape)
    
    return color_code