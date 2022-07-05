### This is the SwissParam webserver ligand force field parametrization code taken from:
### https://github.com/aaayushg/charmm_param/blob/master/swiss.py

from bs4 import BeautifulSoup
from xml.dom import minidom
import mechanize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conf")
args = parser.parse_args()

# initialize the browser
br = mechanize.Browser()
br.set_handle_robots(False)   # ignore robots
br.set_handle_refresh(False)  # can sometimes hang without this
br.addheaders = [('User-agent', 'Firefox.')]
br.set_handle_redirect(mechanize.HTTPRedirectHandler)

# login fill-in form and submit
url = "http://www.swissparam.ch/"
response = br.open(url)
br.form = list(br.forms())[0]

# upload the file to parametrize, parse xml output
filename = args.conf
br.form = list(br.forms())[0]
br.form.add_file(open(filename), 'text/plain', filename)
response = br.submit()
xml = response.read().strip()
#print xml
soup = BeautifulSoup(xml,'html.parser')
for link in soup.find_all('a'):
    if 'swissparam' in link.get('href'):
        print(link.get('href'))
        
#Save links into a folder and download all links using wget
#keep buffer time in wget (for job to complete on webserver)
