## Startup Fail Analysis
In this project, We gonna find out what is the biggest cause of the failed startup. As we all know, money can cause so much trouble, but surprisingly I found that money is not the biggest problem, so what is the number 1?

### **1. Project Objectives**
We are going to answer some Business Question

- What Happened?

According to the BLS, **1/5 startups fail within their first year**. and **up to 65% of startups will fail within 10 years** of opening for businesses.
As a business consultant,

our client wants us to prevent his startup from going fall.
- why did it happen?

1. 38% of startup failures occur due to financial issues
2. 35% of businesses fail due to lack of market need
3. 20% of businesses fail due to industry competition
4. 19% of businesses fail due to unsuccessful business model
5. 18% of businesses fail due to legal challenges

source : https://www.forbes.com/advisor/business/software/startups-failure-rate/

- What will we do?

We're gonna eksplore fail startup dataset to make a predicition model of failing startup. Then giving insight what is the biggest cause of failing startup?

- How we do that?
1. Using Classification Machine learning to consider is the startup will fail (0) or no
2. Evaluation :
  - **False Positive**: Miss predict success startup as a "No"
  - **False Negative**: Miss predict fail startup as a "Success startup"
3. Use Feature Importance and SHAP to find the causes.

### **2. Data Understanding**
- 'Unnamed:0' = possibly representing the index or a placeholder for missing data.
- 'state_code' = state code
- 'latitude' = north south position
- 'longitude' = east west position
- 'zip_code' = postal code
- 'id' = unique values
- 'city'= location
- 'Unnamed: 6' = Another unnamed column, likely a placeholder for missing data or unused information.
- 'name' = startup name
- 'labels' = Maybe same as status
- 'founded_at' = date founded
- 'closed_at' = date closed
- 'first_funding_at' = date first received it's funding
- 'last_funding_at' = The date of the most recent funding round.
- 'age_first_funding_year' = the age of the startup when it received first funding
- 'age_last_funding_year' = The age of the startup in years when it received its last funding.
- 'age_first_milestone_year' = The age of the startup when it achieved its first major milestone.
- 'age_last_milestone_year' = The age of the startup when it achieved its last major milestone.
- 'relationships' = The number of professional relationships or connections the startup has established.
- 'funding_rounds' = The total number of funding rounds the startup has gone through.
- 'funding_total_usd' = The total amount of funding the startup has raised
- 'milestones' = The total number of significant milestones the startup has achieved
- 'state_code.1' = secondary state code, potentially for validation or categorization purposes.
- 'is_CA', 'is_NY', is_MA', 'is_TX' = is it located in california, new york, massachusets or texas.
- 'is_otherstate' = is it located not in one of 4 locations above
- 'category_code' = The industry or category to which the startup belongs
- 'is_software' = in software industry
- 'is_web' = web industry
- 'is_mobile' = mobile industry
- 'is_enterprise' = enterprise industry
- 'is_advertising' = advertising industry
- 'is_gamesvideo' = games or video industry
- 'is_ecommerce' = e-commerce industry
- 'is_biotech' = biotech industry
- 'is_consulting' = consulting industry
- 'is_othercategory' = other than anything above
- 'object_id' =  A unique identifier for each object
- 'has_VC' = the startup has venture capital funding
- 'has_angel' = the startup has received angel funding
- 'has_roundA' = whether the startup has gone through a Series A funding round
- 'has_roundB' = whether the startup has gone through a Series B funding round
- 'has_roundC' = whether the startup has gone through a Series C funding round
- 'has_roundD' = whether the startup has gone through a Series D funding round
- 'avg_participants' = The average number of participants involved in the startup's funding rounds.
- 'is_top500' = is in the top 500 startups?
- 'status' = acquired or closed.

### **3. Exploratory Data Analysis**

**Startup Lifespan Classification**

![Lifespan Startup](https://github.com/user-attachments/assets/e96dd640-c279-4478-aa13-1c0ed14bcde5)

**Findings**🎯
```bash
- 1.75% startups fail in the first year
- 18.34% startups fail in 3 years
- 61.57% startups fail in 6.5 years
- 84.72% startups fail in 10 years
- Average failed startup ages = 6.15%

The data is Left-skewed, with a peak at around 1000–2500 days (about 3–6.5 years). The frequency gradually decreases as the lifespan increases.

- 1-3 years (Adapt Phase)
- 3-6,5 years (Critical Phase)
- '>6,5 years (Maturity Phase)
```


**Total Funding Analysis**

![Funding Total](https://github.com/user-attachments/assets/bf404692-9f36-4e77-98aa-b8fb75e20fb3)

**Findings**🎯
```bash
- The median funding for existing startups is higher than failed startups, This suggests that startups with higher funding tend to survive more often.
- The existing have wider range of funding total, it means some successful startup gain more funding that most failed startups.
- Existing have more extreme funding outliers. The farthest outlier for fail startup just reach the beginning of existing outliers.
```


**Funding Round Analysis**

![Funding Round](https://github.com/user-attachments/assets/a0828383-bbd6-4ff8-a8c4-ef60bfd105d5)

**Findings**🎯
```bash
- 43.7% of startups that received funding have closed, This suggests that while funding can help, analyzing other factors like market fit or execution being critical for survival.
- VC  and Angel funding may not always correlate with long-term survival, possibly due to high-risk investments.
- To improve survivability, startups should aim to advance to later funding stages while balancing growth with operational stability.
```


**Relationship Analysis**

![Relationship](https://github.com/user-attachments/assets/1ea73b51-a275-48d2-876b-3b3f2a4f2204)

**Findings**🎯
```bash
we found that startup that have more relations tend to survive. If a startups have more than 20 Professional Relationship, they will strongly survive! 
```


**Relation v Total Funding**

![Relation v Total Funding](https://github.com/user-attachments/assets/34c74357-17e4-4d7e-aafc-a63e0b110b37)

**Findings**🎯
```bash
Surprisingly, Relations and total funding is not highly correlated. 
```

### **4. Best Modeling**
After training many models, we got the best model and that is :
**Light GBM with SMOTE and Tuning Evaluation**

And here is the Confusion Matrix

![Best Model](https://github.com/user-attachments/assets/dc718853-2273-47dc-a478-a9ad500e2307)

![image](https://github.com/user-attachments/assets/945a3e4a-3cf8-4bfa-84f9-b1025d414908)

![image](https://github.com/user-attachments/assets/f2906e20-c8f0-4e19-a376-6c70f76750fd)


### **5. Feature Importance**

![Variable Importance](https://github.com/user-attachments/assets/80779bcf-e77f-4239-945b-5104d8907f11)

**Findings**🎯
```bash
- Relationship : Relation have the biggest impact to the success or the close of a startup.
- Funding_total_usd : Surprisingly, funding is the next after relationship. And from EDA we don’t see any correlations between them.
- Age last milestone
- is_top500
- age first milestone

Why Relationship?
- Access to Resources : Strong relationships enable startups to obtain funding, mentorship, and relevant technology. Investors not only provide money but also open access to partner or customer networks.
- Reputation and Credibility : Relationships with influential figures (political figures/influencers or artist) can increase market trust and accelerate startup growth.
- Growth Acceleration : Partnering with large companies or institutions can expand market access and facilitate product penetration. Example: Fintech startups that start with large banks gain credibility and customers faster.
```
