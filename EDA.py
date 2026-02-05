import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ================= HELPER FUNCTIONS =================

def add_bar_labels(ax, fmt="%.1f"):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(fmt % height,
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=9)

# ================= EDA CLASS =================

class EDA:

    def __init__(self):
        self.df = pd.read_csv(
            r"C:\Vihara_Tech\Churn_Prediction\Churn_Updated_set.csv"
        )

    def run_visualizations(self):

        # 1️⃣ Churn Distribution (PIE)
        pct = self.df['Churn'].value_counts()
        pct.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Churn Distribution')
        plt.ylabel('')
        plt.show()

        # 2️⃣ Gender-wise Churn (BAR)
        ax = self.df[self.df['Churn'] == 'Yes']['gender'].value_counts().plot(
            kind='bar', title='Gender-wise Churn Count'
        )
        add_bar_labels(ax, fmt="%.0f")
        plt.show()

        # 3️⃣ Churn by Gender & Senior Citizen (BAR)
        churn = self.df[self.df['Churn'] == 'Yes']
        pct = pd.crosstab(churn['gender'], churn['SeniorCitizen'], normalize='index') * 100
        pct.columns = ['Non-Senior', 'Senior']
        ax = pct.plot(kind='bar', title='Churn by Gender & Senior Citizen (%)')
        add_bar_labels(ax)
        plt.show()

        # 4️⃣ Internet Service by Gender (BAR)
        pct = pd.crosstab(self.df['gender'], self.df['InternetService'], normalize='index') * 100
        ax = pct.plot(kind='bar', title='Internet Service by Gender (%)')
        add_bar_labels(ax)
        plt.show()

        # 5️⃣ Phone Service (Churned) (BAR)
        churn['GS'] = churn['gender'] + '-' + churn['SeniorCitizen'].map({0:'Non-Senior',1:'Senior'})
        pct = pd.crosstab(churn['GS'], churn['PhoneService'], normalize='index') * 100
        ax = pct.plot(kind='bar', title='Phone Service among Churned Customers (%)')
        add_bar_labels(ax)
        plt.show()

        # 6️⃣ Multiple Lines by Gender & Senior (BAR)
        self.df['GS'] = self.df['gender'] + '-' + self.df['SeniorCitizen'].map({0:'Non-Senior',1:'Senior'})
        pct = pd.crosstab(self.df['GS'], self.df['MultipleLines'], normalize='index') * 100
        ax = pct.plot(kind='bar', title='Multiple Lines by Gender & Senior (%)')
        add_bar_labels(ax)
        plt.show()

        # 7️⃣ Multiple Lines by SIM (PIE)
        pct = self.df['SIM_Provider'].value_counts()
        pct.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('SIM Provider Distribution')
        plt.ylabel('')
        plt.show()

        # 8️⃣ Multiple Lines by SIM & Gender (BAR)
        self.df['SIM_G'] = self.df['SIM_Provider'] + '-' + self.df['gender']
        pct = pd.crosstab(self.df['SIM_G'], self.df['MultipleLines'], normalize='index') * 100
        ax = pct.plot(kind='bar', title='Multiple Lines by SIM & Gender (%)')
        add_bar_labels(ax)
        plt.show()

        # 9️⃣ Internet Service Distribution (PIE)
        pct = self.df['InternetService'].value_counts()
        pct.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Internet Service Distribution')
        plt.ylabel('')
        plt.show()

        # 🔟 Internet by SIM (BAR)
        ax = pd.crosstab(self.df['InternetService'], self.df['SIM_Provider']) \
            .plot(kind='bar', title='Internet Service by SIM')
        add_bar_labels(ax, fmt="%.0f")
        plt.show()

        # 11️⃣ Internet by SIM & Gender (BAR)
        ax = pd.crosstab(
            [self.df['SIM_Provider'], self.df['gender']],
            self.df['InternetService']
        ).plot(kind='bar', title='Internet by SIM & Gender')
        add_bar_labels(ax, fmt="%.0f")
        plt.show()

        # 12️⃣ Internet by SIM, Gender & Churn (BAR)
        ax = pd.crosstab(
            [self.df['SIM_Provider'], self.df['gender']],
            self.df['Churn']
        ).plot(kind='bar', title='Internet by SIM, Gender & Churn')
        add_bar_labels(ax, fmt="%.0f")
        plt.show()

        # 13️⃣ Service Distribution (BAR – 6 plots)
        services = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies']
        for s in services:
            pct = self.df[s].value_counts(normalize=True) * 100
            ax = pct.plot(kind='bar', title=f'{s} Distribution (%)')
            add_bar_labels(ax)
            plt.show()

        # 19️⃣ Contract Distribution (PIE)
        pct = self.df['Contract'].value_counts()
        pct.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Contract Distribution')
        plt.ylabel('')
        plt.show()

        # 20️⃣ Paperless Billing (BAR)
        ax = pd.crosstab(self.df['SIM_Provider'], self.df['PaperlessBilling']) \
            .plot(kind='bar', title='Paperless Billing by SIM')
        add_bar_labels(ax, fmt="%.0f")
        plt.show()

        # 21️⃣ Payment Method by Gender & Churn (BAR)
        ax = pd.crosstab(
            [self.df['PaymentMethod'], self.df['gender']],
            self.df['Churn']
        ).plot(kind='bar', title='Payment Method by Gender & Churn')
        add_bar_labels(ax, fmt="%.0f")
        plt.xticks(rotation=45)
        plt.show()

        # 22️⃣ Tenure w.r.t Churn (BAR)
        ax = self.df.groupby('Churn')['tenure'].mean().plot(
            kind='bar', title='Average Tenure by Churn'
        )
        add_bar_labels(ax)
        plt.show()


# ================= RUN =================

eda = EDA()
eda.run_visualizations()
