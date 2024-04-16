# GDELT-Conflict-Exploration
This is an exploration of the GDELT dataset and hypothesis testing for, Event Frequency Comparison, and Temporal Analysis


'''
GDELT 1979-2021

Columns in the GDELT Dataset.

Year: The year in which the events occurred.
CountryCode: The code representing the country where the event took place.
CountryName: The name of the country where the event took place.
SumEvents: Count of events by event code, year, and country.
TotalEvents: Total events by year and country according to the GDELT 1.0 normalization files.
NormalizedEvents1000: Proportion of events on the normalization total (over 1000), calculated as SumEvents/TotalEvents * 1000.
EventRootCode: Conflict and Mediation Event Observations (CAMEO) event root code, representing the general category or root of the event.
EventRootDescr: Description of the CAMEO event root, providing more context about the general category of the event.
EventCode: CAMEO event code, representing the specific event within the general category.
EventDescr: Description of the CAMEO event, providing more detailed information about the specific event.
GoldsteinScale: Goldstein scale value indicating the potential impact of an event on political stability (-10 to +10).
AvgNumMentions: Average number of mentions of the event in news articles.
SumNumMentions: Sum of mentions of the event in news articles.
AvgAvgTone: Average tone of news articles mentioning the event.

Categorical Fields:

CountryCode: Represents the code identifying the country where the event occurred.

CountryName: Represents the name of the country where the event occurred.

EventRootCode: Represents the broad category or root of the event according to the CAMEO coding scheme.

EventRootDescr: Provides a description of the CAMEO event root.

EventCode: Represents the specific event within the broader category.

EventDescr: Provides a description of the specific event.

There are 6 categorical fields in total, representing country codes, country names, and event categories.

Quantitative Fields:

Year: Represents the year in which the events occurred.

SumEvents: Represents the count of events by event code, year, and country.

TotalEvents: Represents the total events by year and country according to the GDELT 1.0 normalization files.

NormalizedEvents1000: Represents the proportion of events on the normalization total, adjusted for comparison purposes.

GoldsteinScale: Represents the Goldstein scale value indicating the potential impact of an event on political stability.

AvgNumMentions: Represents the average number of mentions of the event in news articles.

SumNumMentions: Represents the sum of mentions of the event in news articles.

AvgAvgTone: Represents the average tone of news articles mentioning the event.

There are 8 quantitative fields in total, representing various counts, scores, and averages related to events.

Missing Data or Data Anomalies:

Missing data could exist, particularly in fields such as "GoldsteinScale," "AvgNumMentions," "SumNumMentions," and "AvgAvgTone," where certain events may not have been mentioned in news articles or where data collection may not have been comprehensive.

Summarization of Data:

The observations in the dataset appear to be raw and unprocessed, containing detailed information about individual events such as event codes, descriptions, counts, and scores.

Summarization may have been performed for certain fields, such as calculating averages or sums, but the observations themselves seem to be at the event level rather than aggregated or summarized.


Trend Analysis:

How have the frequency and distribution of events changed over time?
Are there any notable trends or patterns in event occurrences across different years or regions?
What factors might be driving changes in event frequencies, and how might these trends impact decision-making processes?
Geospatial Analysis:

Which countries experience the highest number of events, and are there any regional hotspots or areas of heightened activity?
How do event frequencies vary across different regions, and are there any spatial patterns or clusters of events that warrant further investigation?
Are there correlations between event occurrences and geopolitical factors such as conflicts, political instability, or economic conditions?
Event Categorization and Impact Analysis:

What are the most common types of events recorded in the dataset, and how do they differ in terms of frequency, impact, and tone?
Are certain types of events associated with higher levels of political instability or conflict escalation, as indicated by the GoldsteinScale values?
How do variations in event characteristics (e.g., number of mentions, average tone) correlate with their perceived impact or significance?
Media Coverage and Public Perception:

How does media coverage of events vary across different countries and regions?
Are there discrepancies between the frequency or tone of events reported in the media and their actual occurrence or impact on the ground?
What factors influence the extent of media coverage for certain events, and how might this affect public perception and decision-making?
Predictive Modeling and Forecasting:

Can historical event data be used to develop predictive models for anticipating future events or identifying potential areas of concern?
Are there any early warning indicators or precursors to certain types of events that could inform proactive intervention or risk mitigation strategies?


Hypothesis: There is a significant difference in the average number of events between countries with high and low levels of political stability.

Null Hypothesis (H0): The average number of events is the same for countries with high and low levels of political stability.
Alternative Hypothesis (H1): The average number of events differs between countries with high and low levels of political stability.
Hypothesis: There is a significant difference in the average Goldstein scale values between different types of events.

Null Hypothesis (H0): The average Goldstein scale value is the same for all types of events.
Alternative Hypothesis (H1): The average Goldstein scale value varies across different types of events.
Hypothesis: Events categorized as "FIGHT" have a higher average number of mentions in news articles compared to events categorized as "COERCE."

Null Hypothesis (H0): There is no difference in the average number of mentions between "FIGHT" and "COERCE" events.
Alternative Hypothesis (H1): "FIGHT" events have a higher average number of mentions than "COERCE" events.
Hypothesis: There is a significant difference in the average tone of news articles mentioning events categorized as "FIGHT" and "COERCE."

Null Hypothesis (H0): There is no difference in the average tone of news articles mentioning "FIGHT" and "COERCE" events.
Alternative Hypothesis (H1): "FIGHT" events have a different average tone in news articles compared to "COERCE" events.
Hypothesis: The average number of events has changed significantly over time.

Null Hypothesis (H0): There is no significant difference in the average number of events across different years.
Alternative Hypothesis (H1): The average number of events has changed over time.
Hypothesis: Events occurring in certain regions have a different average Goldstein scale value compared to events in other regions.

Null Hypothesis (H0): There is no difference in the average Goldstein scale value across different regions.
Alternative Hypothesis (H1): The average Goldstein scale value varies across different regions.


Predictive Linear Regression:

"I would like to predict the average number of events (SumEvents) based on the features such as the Goldstein scale, average number of mentions, and average tone of news articles (GoldsteinScale, AvgNumMentions, AvgAvgTone)."
Inferential Linear Regression:

"I would like to use inferential regression to gain insight into how the Goldstein scale, average number of mentions, and average tone of news articles (GoldsteinScale, AvgNumMentions, AvgAvgTone) affect the frequency of events (SumEvents)."
Predicting a Categorical Variable with Logistic Regression:

"I would like to predict whether an event is likely to be categorized as 'FIGHT' or 'COERCE' based on features such as the Goldstein scale, average number of mentions, and average tone of news articles (GoldsteinScale, AvgNumMentions, AvgAvgTone)."
'''