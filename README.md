# Six-Nations-Correlation-with-World-Cup-Performance
**[Link to blog](LINK)**


## Objective:
This project is designed to scrape and analyse data to explore the relationships between performance in the Six Nations Championship and final standings in the Rugby World Cup. 

Additionally, our analysis includes an examination of investment data derived from the annual statements of relevant parties, offering a comprehensive look at how financial aspects influence the sport.

Given the unique nature of this task, the data is gathered from multiple sources, each offering a distinct perspective on the world of rugby. 

The objective of this blog post is to provide insights that are informative and accessible, not only to rugby enthusiasts but to a broader audience as well. 

## Project Structure:
1. Data Collection (Web Scraping) and Form Clean Dataframes
2. Six Nations Data Analysis
3. Investment Analysis
4. World Cup Quarter Finalists (Top 8) Analysis
   * Visualisation of Quarterfinals Apperences of Six Nations Teams
   * Correlation between Six Nations performance and reaching World Cup Quarterfinals
6. Average performances in Six Nations to reach World Cup Quarterfinals and Top 3
   * Total Points
   * Points Scored
8. Correlation between Six Nations Winner and reaching World Cup Top 3

## Setup Instructions
### Running the Project
The project files are located within the live-script folder and are designed to be run in a Python 3 environment using Jupyter Notebook through Anaconda.

**Open Anaconda Navigator:**
Launch the Anaconda Navigator application and open Jupyter Notebook.

**Navigate to the Project:**
Within Jupyter Notebook, navigate to the live-script folder and open the Empirical_Project.ipynb file.


### Installing Required Packages
Before running the notebook, ensure all the necessary Python packages are installed. Here are the installation instructions for the required packages:

**For Web Scraping:**
```python
%pip install beautifulsoup4 requests selenium bs4
%pip install --upgrade selenium webdriver-manager
```

**For Analysis and Visualization:**
```python
%pip install notebook graphviz pearsonr statsmodels ipywidgets
%pip install plotly --upgrade
```
#Relevent installs and updates included in working script


### Key Libraries Used
* **Pandas**: Used for data manipulation and analysis, ideal for working with structured data.

* **NumPy**: Essential for handling large, multi-dimensional arrays and matrices, and a wide range of mathematical operations.

* **Seaborn**: A statistical data visualization library based on matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.

* **Pyplot**: Part of matplotlib, helps in creating static, interactive, and animated visualizations in Python.

#Librarys are imported when needed throughout the script


### Data Sets Used:
In this project the majority of the data is found through scraping the web. However, in the Investment Analysis section the data was manualy extracted and compiled into the file '6_nations_teams_investments.csv' located in the [essential_datasets](https://github.com/Connor-g-c/Six-Nations-Correlation-with-World-Cup-Performance/tree/main/Datasets/essential_datasets) folder. You must make sure this file is downloaded to your local PC, and then upload it to you Jupyter environment so that the file may be read in the working script.


## References:
1. [Annual Reports - Scottish Rugby. (2023, November 6). Scottish Rugby.](
     https://scottishrugby.org/about/annual-reports/)
2. [Documenti - Federazione Italiana Rugby. (2023, October 16). Federazione Italiana Rugby.](
   https://federugby.it/fir-documenti/)
3. [EUR/GBP Currency Exchange Rate (0.86 on 02/05/2024) Google Finance.](
    https://www.google.com/finance/quote/EUR-GBP?sa=X&sqi=2&ved=2ahUKEwjGg4eQ-_2FAxWzRkEAHXYuBkYQmY0JegQIFBAw)
4. [Fédération Française de Rugby. (2024).](                  
    https://www.ffr.fr/ffr/publications-officielles/financier)
5. [Irish Rugby | Annual Report. (2016). Irishrugby.ie.](
    https://www.irishrugby.ie/irfu/about/annual-report/)
6. [RFU. (2024). Englandrugby.com.](
    https://www.englandrugby.com/about-rfu/annual-reports)
7. [Reports - Welsh Rugby Union | Club & Community. (2024, May 7)](
    https://community.wru.wales/the-wru/reports/)
8. [Rugbypass. (2023). Rugbypass.com](
    https://www.rugbypass.com/six-nations/info-and-faq/rules/)
9. [Six Nations Results Archive / Database | Livesport.com. (2024). Livesport.com](
    https://www.livesport.com/en/rugby-union/europe/six-nations/archive/)
10. [Wikipedia Contributors. (2024, April 23). Rugby World Cup. Wikipedia](
    https://en.wikipedia.org/wiki/Rugby_World_Cup)
