class Taipei:
    lat = 25.0394
    lon = 121.5066

class Yushan:
    lat = 23.4894
    lon = 120.9514

class BaoZhong:
    lat = 23.7029
    lon = 120.3029

class Taoyuan:
    lat = 25.0085
    lon = 121.0393

class Miaoli:
    lat = 24.4189
    lon = 120.8664

class Hsinchu:
    lat = 24.8296
    lon = 121.0060

class Taichung:
    lat = 24.1475
    lon = 120.6759

class Liga:
    lat = 23.3949
    lon = 120.7066

class Nantou:
    lat = 23.9692
    lon = 120.7942

class Kaohsiung:
    lat = 22.6251
    lon = 120.2675

class Pingtung:
    lat = 22.7650
    lon = 120.6964

class Chaozhou:
    lat = 22.5361  
    lon = 120.5320

area = {'Taipei': Taipei,
        'Yushan': Yushan,
        'BaoZhong': BaoZhong,
        'Taoyuan': Taoyuan,
        'Miaoli': Miaoli,
        'Hsinchu': Hsinchu,
        'Taichung': Taichung,
        'Liga': Liga,
        'Nantou': Nantou,
        'Pingtung': Pingtung,
        'Kaohsiung': Kaohsiung,
        'Chaozhou': Chaozhou}

#466880 - 板橋 466880  板橋         0   24.9994  121.4338
#466940 - 基隆466940  基隆         0   25.1351  121.7322 
#466920 - 臺北466920  臺北         0   25.0394  121.5066
#467050 - 新屋467050  新屋         0   25.0085  121.0393
#467480 - 嘉義467480  嘉義         0   23.4977  120.4248
#467540 - 大武467540  大武         0   22.3574  120.8957
#467590 - 恆春467590  恆春         0   22.0057  120.7383 
#467610 - 成功467610  成功         0   23.0993  121.3653
#467650 - 日月潭467530  阿里山       0   23.5100  120.8051
#467660 - 臺東467660  臺東         0   22.7540  121.1465
#467770 - 梧棲467770  梧棲         0   24.2578  120.5152
#466990 - 花蓮466990  花蓮         0   23.9769  121.6051 
#C0G840 - 北斗C0G840  北斗         0   23.8792  120.5065 
#C0K430 - 褒忠C0K430  褒忠         0   23.6927  120.2955
#C0R220 - 潮州C0R220  潮州         0   22.5361  120.5320 
#C0R550 - 新埤C0R550  新埤         0   22.4698  120.5748
#C0U990 - 員山
#C0Z061 - 玉里C0Z061  玉里         0   23.3233  121.3316
#C1F941 - 雪嶺C1F941  雪嶺         0   24.2825  121.0186
#C1R120 - 上德文C1R120  上德文       0   22.7650  120.6964


class area_20_466880:#板橋
    lat =  24.9994
    lon = 121.4338
class area_20_466940:#基隆
    lat =  25.1351
    lon = 121.7322 
class area_20_466920:#臺北
    lat =  25.0394
    lon = 121.5066  
class area_20_467050:#新屋
    lat =  25.0085
    lon = 121.0393
class area_20_467480:#嘉義
    lat =  23.4977
    lon = 120.4248 
class area_20_467540:#大武
    lat =  22.3574  
    lon = 120.8957
class area_20_467590:#恆春
    lat =   22.0057  
    lon = 120.7383
class area_20_467610:#成功
    lat =  23.0993   
    lon = 121.3653    
class area_20_467650:#日月潭
    lat =   23.8831 
    lon = 120.8999 
class area_20_467660:#臺東
    lat =  22.7540  
    lon = 121.1465 
class area_20_467770:#梧棲
    lat =  24.2578 
    lon = 120.5152
class area_20_466990:#花蓮
    lat =  23.9769   
    lon =  121.6051 

class area_20_C0G840:#北斗
    lat =  23.8792 
    lon =  120.5065 
class area_20_C0K430:#褒忠
    lat = 23.6927
    lon =  120.2955
class area_20_C0R220:#潮州
    lat =  22.5361 
    lon = 120.5320     
class area_20_C0R550:#新埤
    lat = 22.4698
    lon =   120.5748
class area_20_C0U990 :#員山
    lat = 24.7461 
    lon = 121.7143
class area_20_C0Z061:#玉里
    lat = 23.3233 
    lon =  121.3316
class area_20_C1F941 :#雪嶺
    lat = 24.2825 
    lon = 121.0186    
class area_20_C1R120 :#上德文
    lat = 22.7650  
    lon = 120.6964 
    '''
    466880 - Banqiao
466940 - Keelung
466920 - Taipei
467050 - New_House
467480 - Chiayi
467540 - Dawu
467590 - Hengchun
467610 - Success
467650 - Sun_Moon_Lake
467660 - Taitung
467770 - 梧栖
466990 - Hualien
C0G840 - Beidou
C0K430 - 褒忠
C0R220 - Chaozhou
C0R550 - News
C0U990 - Member_Hill
C0Z061 - Yuli
C1F941 - Snow_Ridge
C1R120 - Shangdewen
    '''
area_20={
        'Banqiao':area_20_466880,
        'Keelung':area_20_466940,
        'Taipei':area_20_466920,
        'New_House':area_20_467050,
        'Chiayi':area_20_467480,
        'Dawu':area_20_467540,
        'Hengchun':area_20_467590,
        'Success':area_20_467610,
        'Sun_Moon_Lake':area_20_467650,
        'Taitung':area_20_467660,
        'Yuxi':area_20_467770,
        'Hualien':area_20_466990,
        'Beidou':area_20_C0G840,
        'Bao_Zhong':area_20_C0K430,
        'Chaozhou':area_20_C0R220,
        'News':area_20_C0R550,
        'Member_Hill':area_20_C0U990,
        'Yuli':area_20_C0Z061,
        'Snow_Ridge':area_20_C1F941,
        'Shangdewen':area_20_C1R120,
        }