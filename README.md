## Background
Hire Heroes USA empowers U.S. military members, veterans, and spouses to succeed in the civilian workforce.  Through our free, signature workshops and online programs, clients are individually partnered with a highly trained Veteran Transition Specialist who works with the client to: create a tailored civilian resume that effectively highlights skills and achievements, translate military experience into civilian terminology, learn effective job searching, networking and interviewing techniques, and get connected with companies that actively hire veterans.  

As a result, Hire Heroes USA confirms hired on average more than 60 clients per week.  Though this success has allowed growth in both scope and impact, there are still roughly 500,000 unemployed veterans that exist at any given time.  In order to better reach this population, Hire Heroes USA would like to analyze existing client, social media, and volunteer data to determine if there are opportunities for further improvements to our systems.

## Context:

Though all donations are updated in Salesforce, individual donations are done through a tool called Classy. Some donations are grouped as part of a campaign, while others are part of grants or corporate giving programs.Email campaigns are completed by using a software tool called Vertical Response.

* The data from this is included (as VREmailHistory), and anything identified as Top Jobs or Virtual Career Fair in Targeted Email Subject should be helpful here.

* Questions:
  1.  Is there a geographic location within the US that most of our individual donors come from? Are there areas in the country we don't see any donors from?
  2. Do our social media posts or fundraisers calling for donations hit these areas with little to no donors?
  3. Do email campaigns have any effect on individual donations?
  4. Do email campaigns have any effect on job seekers creating profiles on the Hire Heroes USA Job Board?
  5. Is there a relationship between certain days of the week, times in the day, or months, or time of year and when employers and job se


### EDA
1. We explored Opporotunity CSV

  
  - **Campaigns**
    - Found the major Campaings contributing to most of the revenue
      - <img src="https://user-images.githubusercontent.com/6872080/78728506-8ae0b100-7905-11ea-9337-d3727fbc656c.png" height="200">

    - Grants contribute disproportionatly larger than individual owners or any other source
    - We also looked at the Number of Donors with various different Donation types like - Event, Individual , Coroporate Etc.
      - <img src="https://user-images.githubusercontent.com/6872080/78728579-b368ab00-7905-11ea-82db-e55ab6b4918b.png" height="300">
    - Even though there are a large number of campaigns and donations from individuals but grants top the chart.
    - Even though there are a large number of campaigns only 4-5 bring most of the money
      - <img src="https://user-images.githubusercontent.com/6872080/78728634-d98e4b00-7905-11ea-9bb7-8672063718bc.png" height="300">
    - Top one being the AAA
      - We also see that AAA campaign is the only campaign that involves all types of campaign types including - Ticketed,Donations,Peer to peer etc
    - We see from campaigns.csv that most of the data available is categorical in nature making it hard to do predictive modelling like linear regression.
    - A lot of data about the campaigns is missing or mostly just 0
    
  - **Trend**
    - We also see an increasing trend in the total amount of the donations going up till 2018 and then there is a slight drop in 2019- maybe due to data being less for 2019
    - Amount secured in grants shot up in 2014
    - Most of the donations have been secured on **National Veterans' Day**
    - <img src="https://user-images.githubusercontent.com/6872080/78728720-13f7e800-7906-11ea-8584-924c4cf3c38f.png" height="300">
    - <img src="https://user-images.githubusercontent.com/6872080/78728679-f4f95600-7905-11ea-8bb5-4d600c7a99a2.png" height="300">
   
  - **Location**
    - We see that most of the campaigns were held in VA,CA,TX,GA and NC and specially in locations like Fort * 
    - There are also a number of virtual campaigns
    - <img src="https://user-images.githubusercontent.com/6872080/78728752-2f62f300-7906-11ea-959c-f39989d5c0ad.png" height="300">
    
  - **Sponsors**
    - Looking at the sponsers we see that most of the Workshops have been sponsored by - USO -  United Service Organizations
    - Workshops mostly deal with Career Transition followed by employment workshops
    - HIUSA - Even though they don't do many workshops but they hold a variety of different types of workshops 
     - <img src="https://user-images.githubusercontent.com/6872080/78728830-6df8ad80-7906-11ea-8c4c-9e6a73c04460.png" height="300">
     
  - **Campaign Duration**
    - Most of the campaigns are short lived and last for a few days
      - These include mostly the fundraisers or Short immediate Grants
      - We see that donations and peer to peer campaigns are the longest ones as they usually don't have an end date.
      - <img src="https://user-images.githubusercontent.com/6872080/78728874-88328b80-7906-11ea-975b-7f37b5e0492e.png" height="400">
      
  - **Contacts**
    - Applicants
      - 2018 saw the highest number of applicants followed by 2016
      - <img src="https://user-images.githubusercontent.com/6872080/78729000-e2335100-7906-11ea-9b8e-5d3d19fd9e8b.png" height="300">
      - <img src="https://user-images.githubusercontent.com/6872080/78729073-13138600-7907-11ea-90fb-09995508b3d7.png" height="300">

### Modeling
* Since we found that **most of the data we have is categorical**, it makes it harder to build predictive models specially regression models on variable of interest like Amount secured in a campaign.
* Also there are variables like expectedrevenue etc which are **highly correlated** and therefore lead to biased predictions.
  - On removing such variables the r^2 values drop considerably from around .97 to 0.002 or even negative for a cross validated model selection.
* We might **try clustering** but again the number of categorical variables makes using common distance metrics like eucleadian distances very difficult.


## Folder structure

* ![image](https://user-images.githubusercontent.com/6872080/76349690-db51f800-62e0-11ea-89f1-431070a42e31.png)
* Extract the zip to GivenContent Folder and then extract each of the data zip files to a folder inside GivenContent/Data
