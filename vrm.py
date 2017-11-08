#!/usr/bin/python -u
# *-------------------------------------------------------------------------
# * Title:   vrm-bbb.py
# *
# * Author:  Gilbert
# *
# * Purpose:
# * A program which listens for button presses on a GPIO pin & GPS then it will
# * reports(send) to VRM Server.
# --------SGL_Host: 52.76.64.132:3010 [38, 39, 35]

import os, select, sys, json, datetime,  socket, fnmatch, time, subprocess, urllib
# , MySQLdb

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
# import Adafruit_GPIO.SPI as SPI , Adafruit_SSD1306, Image, ImageDraw , ImageFont
from stat import *
#from adxl345 import ADXL345

# from time import sleep


print
"------------------------------------Step 0:Begin"
print
"@@   @@@@@@@@@@    @@@@@@@@     @@"
print
"@@       @@        @@    @@     @@"
print
"@@       @@        @@@@@@@@     @@"
print
"@@       @@        @@   @@@     @@"
print
"@@       @@        @@    @@@    @@"
print
"------------------------------------Step 0:End"
print
"Beginning: for debug"


def cat(f):
    with open(f, 'r') as fobj:
        return fobj.read()


def trycat(f, defval):
    if os.path.isfile(f):
        return cat(f)
    else:
        return defval


VIDEO_LOG = "/root/video-log.txt"
GPS_LOG = "/root/gps-log.json"

SGL_HOST = "52.76.64.132:3010"

# HOST_ID='/boot/uboot/host-id.txt'
# SGL_HOST=trycat(HOST_ID,'52.76.64.132:3010')
CONF_ID = '/boot/uboot/sgl-id.txt'

# SGL_ID=cat(CONF_ID)
SGL_ID = 'ABC-123'
# int(trycat(CONF_ID, 1))

CONF_PERIOD = '/boot/uboot/sgl-period.txt'
SGL_PERIOD = int(trycat(CONF_PERIOD, 10))
GPIO_PINS = [67, 68, 69]


# def oled_set(time1,key1,val1,key2,val2,key3,val3,val4):
# Beaglebone Black pin configuration:
#    RST = 'P9_12'
# Note the following are only used with SPI:
#    DC = 'P9_15'
#    SPI_PORT = 1
#    SPI_DEVICE = 0
#    disp = Adafruit_SSD1306.SSD1306_128_64(rst=RST, dc=DC, spi=SPI.SpiDev(SPI_PORT, SPI_DEVICE, max_speed_hz=8000000))

#    print "OLED start"
# Initialize library.
#    disp.begin()
# Clear display.
#    disp.clear()
#    disp.display()

# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
#    width = disp.width
#    height = disp.height
#    image = Image.new('1', (width, height))

# Get drawing object to draw on image.
#    draw = ImageDraw.Draw(image)

# Draw a black filled box to clear the image.
#    draw.rectangle((0,0,width,height), outline=0, fill=0)

# Draw some shapes.
# First define some constants to allow easy resizing of shapes.
#    padding = 1
#    shape_width = 0
#    top = padding
#    bottom = height-padding

# Move left to right keeping track of the current x position for drawing shapes.
#    x = padding

# Load default font.
#    font = ImageFont.load_default()

# Write line1 of text.
#    draw.text((x, top),    'T:',  font=font, fill=255)
#    draw.text((x +10 , top), '%s'%time1, font=font, fill=255)

# line5
#    draw.text((x, top +40),    '%s:'%key1,  font=font, fill=255)
#    draw.text((x+25, top +40), '%.2f'%val1,  font=font, fill=255)
#    draw.text((x +60 , top +40), '%s:'%key2, font=font, fill=255)
#    draw.text((x +85 , top +40), '%.2f'%val2 , font=font, fill=255)
# line6
#    draw.text((x , top +50),    'Vf:%s'%val4 , font=font, fill=255)

#    if(val3):
# line4
#        draw.text((x  , top +30), '%s:'%key3, font=font, fill=255)
#        draw.text((x +50 , top +30), '%s'%val3 , font=font, fill=255)
#    else:
#        draw.text((x , top +30),'No Event' , font=font, fill=255)


# Display image.
#    disp.image(image)
#    disp.display()


def find(base, patt):
    for d in os.listdir(base):
        if fnmatch.fnmatch(d, patt):
            return base + "/" + d
    return ""


def vrm_bbb_gps_log():
    # {"class":"TPV","tag":"MID2","device":"/dev/gps0","mode":2,"time":"2014-11-28T09:56:03.560Z","ept":0.005,"lat":24.967768304,"lon":121.546312048,"track":233.1851,"speed":0.511}
    proc = subprocess.Popen(['/usr/bin/gpspipe', '-w'], stdout=subprocess.PIPE)
    print
    "pipe created"
    while True:
        line = proc.stdout.readline()
        parsed = json.loads(line)
        try:
            echo(json.dumps([parsed["lat"], parsed["lon"]]), GPS_LOG)
        except KeyError:
            pass


def gps_read():
    if os.path.isfile(GPS_LOG):
        with open(GPS_LOG, 'r') as f:
            return json.load(f)
    else:
        return [10, 10]


def curl(url, data):
    print
    '{} <- {}'.format(url, data)
    req = urllib2.Request(url, data, {'Content-Type': 'application/json'})
    try:
        f = urllib2.urlopen(req)
        ret = f.read()
        print
        "response" + ret
        f.close()
        return ret
    except urllib2.HTTPError as e:
        return e


def echo(s, f):
    with open(f, 'w') as fobj:
        fobj.write(s)


def gpio_dir(pin):
    return "/sys/class/gpio/gpio" + str(pin)


def gpio_setup(pin):
    d = gpio_dir(pin)
    if not os.path.isdir(d):
        echo(str(pin), "/sys/class/gpio/export")

    echo("in", d + "/direction")
    echo("both", d + "/edge")
    echo("1", d + "/active_low")
    return d


def gpio_val(pin):
    return cat(gpio_dir(pin) + "/value")


def gpio_poll(pins):
    poll = select.poll()
    opened = []
    for p in pins:
        f = open(gpio_dir(p) + "/value")
        opened.append(f)
        poll.register(f, select.POLLPRI | select.POLLERR)

    poll.poll()
    for f in opened:
        f.seek(0)
        f.read()

    poll.poll()
    ret = []
    for f in opened:
        f.seek(0)
        ret.append(int(f.read()))
        poll.unregister(f)
        f.close()
    return ret


def gen_lack(vals):
#    filtered = filter(lambda (k, v): v is 1, enumerate(vals))
#    return ','.join(map(lambda (k, v): str(k), filtered))
     return True

def vrm_bbb_gpio():
    vals = map(lambda p: int(cat(gpio_setup(p) + "/value").strip()), GPIO_PINS)
    sgl_post('plack', 'value', gen_lack(vals))
    while True:
        sgl_post('plack', 'value', 'Traffic-Jam' + gen_lack(gpio_poll(GPIO_PINS)))


def vrm_bbb_ain():
    ocp = find("/sys/devices", "ocp.*")
    helper = find(ocp, "helper.*")
    if not helper:
        bone_capemgr = find("/sys/devices", "bone_capemgr.*")
        echo("cape-bone-iio", bone_capemgr + "/slots")
        helper = find(ocp, "helper.*")

    while True:
        val = int(cat(helper + "/AIN0"))
        sgl_post('pinfo', "battery_info", "{}%".format(val * 100 / 1800.0))
        time.sleep(SGL_PERIOD)


def vrm_bbb_adxl_345():
    adxl345 = ADXL345()
    myfilename = "accel.dat"
    with open(myfilename, 'w') as f:
        print
        "ADXL345 on address 0x%x:" % (adxl345.address)
        f.write(str("#X,Y,Z	ADXL345 accelerometer readings\n"))
        try:
            while (1):
                axes = adxl345.getAxes(True)
                xVal = axes['x']
                yVal = axes['y']
                zVal = axes['z']

                print
                "x = %.3fG " % (xVal)
                print
                "y = %.3fG " % (yVal)
                print
                "z = %.3fG \n" % (zVal)

                f.write(str(xVal) + ",")
                f.write(str(yVal) + ",")
                f.write(str(zVal) + '\n')
                time.sleep(1)
        except KeyboardInterrupt:
            exit()


def sgl_post(target, key, val):
    print
    "response:" + curl('http://' + SGL_HOST + '/meta/post', sgl_json(key, val))


def sgl_json(key, val):
    lat, lng = gps_read()

    vfile = video_read()

    dt = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S+0800')

    # Display image.
    LAT = 'Lat'
    LNG = 'Lng'
    EVENT = 'Event'
    #    oled_set(dt,LAT,lat,LNG,lng,EVENT,val,'')

    return json.dumps({'carId': SGL_ID,
                       'data': [{'dateTime': dt, 'type': 'gps', 'value': {'type': 'Point', 'coordinates': [lng, lat]}},
                                {'dateTime': dt, 'type': 'event', key: val}], 'videos': []})


def vrm_bbb_video(top, callback):
    '''recursively descend the directory tree rooted at top,
       calling the callback function for each regular file'''

    for f in os.listdir(top):
        pathname = os.path.join(top, f)
        mode = os.stat(pathname)[ST_MODE]
        if S_ISDIR(mode):
            # It's a directory, recurse into it
            vrm_bbb_video(pathname, callback)
        elif S_ISREG(mode):
            # It's a file, call the callback function
            print(f)
            echo(f, VIDEO_LOG)
            vfile = f.split('_')[0] + '_' + f.split('_')[1] + '_ch1.avi'
            jfile = f.split('_')[0] + '_' + f.split('_')[1] + '_ch1.jpg'
            print
            "response:" + curl('http://' + SGL_HOST + '/meta/post', sgl_json_video('id', vfile, 'idj', jfile))
            os.system("ffly video1 snapshot /root/sdcard/%s" % jfile)
            callback(pathname)
        else:
            # Unknown file type, print a message
            print('Skipping %s' % pathname)


def visitfile(file):
    time.sleep(1)


def video_read():
    if os.path.isfile(VIDEO_LOG):
        with open(VIDEO_LOG, 'r') as f:
            return f.read()


def video_log(top, callback):
    uCount = 1
    while True:
        vrm_bbb_video(top, callback)
        time.sleep(150)
        print
        "video_log() while loop: %d" % uCount
        uCount = uCount + 1


def sgl_json_video(key, val, key1, val1):
    lat, lng = gps_read()
    vfile = video_read()
    dt = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S+0800')
    # Display image.
    LAT = 'Lat'
    LNG = 'Lng'

    #    oled_set(dt,LAT,lat,LNG,lng,'','',vfile)

    return json.dumps({'carId': SGL_ID,
                       'data': [{'dateTime': dt, 'type': 'gps', 'value': {'type': 'Point', 'coordinates': [lng, lat]}},
                                {'dateTime': dt, 'type': 'event', 'value': 'Non-Event'}],
                       'videos': [{'dateTime': dt, key: val, key1: val1}]})


sname = os.path.basename(__file__)

if sname == "vrm-bbb-gpio.py":
    vrm_bbb_gpio()
elif sname == "vrm-bbb-adxl345.py":
    vrm_bbb_adxl_345()
elif sname == "vrm-bbb-video.py":
    video_log(sys.argv[1], visitfile)
elif sname == "vrm-bbb-ain.py":
    vrm_bbb_ain()
elif sname == "vrm-bbb-gpslog.py":
    vrm_bbb_gps_log()