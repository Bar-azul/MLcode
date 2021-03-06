import driver as driver
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

def crawling():

    url='https://www.homeless.co.il/sale/'
    path = "C:\Program Files (x86)\chromedriver.exe"
    driver = webdriver.Chrome(path)

    types = []
    cities = []
    nhoods = []
    streets = []
    rooms = []
    floors = []
    prices = []

    for page in range(1, 300):
        driver.get(url+str(page))
        matches = driver.find_element(by=By.XPATH, value='//table[@id="mainresults"]//tr[@onclick]')


        type = driver.find_elements(by=By.XPATH, value='//table[@id="mainresults"]//tr[@onclick]//td[3]')
        city = driver.find_elements(by=By.XPATH, value='//table[@id="mainresults"]//tr[@onclick]//td[4]')
        nhood = driver.find_elements(by=By.XPATH, value='//table[@id="mainresults"]//tr[@onclick]//td[5]')
        street = driver.find_elements(by=By.XPATH, value='//table[@id="mainresults"]//tr[@onclick]//td[6]')
        room = driver.find_elements(by=By.XPATH, value='//table[@id="mainresults"]//tr[@onclick]//td[7]')
        floor = driver.find_elements(by=By.XPATH, value='//table[@id="mainresults"]//tr[@onclick]//td[8]')
        price = driver.find_elements(by=By.XPATH, value='//table[@id="mainresults"]//tr[@onclick]//td[9]')

        for i in range(len(price)):
            types.append(type[i].text)
            cities.append(city[i].text)
            nhoods.append(nhood[i].text)
            streets.append(street[i].text)
            rooms.append(room[i].text)
            floors.append(floor[i].text)
            prices.append(price[i].text)
  
    df = pd.DataFrame({ 'type':types,'price': prices,'room':rooms,'floor':floors,'nhood':nhoods,'street':streets, 'city':cities})
    df.to_csv('Houses.csv', index=False)
    driver.quit()
    return df
