#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Project Title: Enterprise Cybersecurity Network Traffic Analysis


# In[ ]:


https://github.com/jonkeaves-cloud/DataAnalytics26.git


# In[ ]:


# This data source is from https://www.kaggle.com/datasets/dhrubangtalukdar/cybersecurity-threat-detection-dataset, and the reason I chose this dataset
# is that I have an Interest in entering the Cybersecurity field of study. This data was collected from September 2025 to December 2025 and features 
# multiple sources of information involving common ports that are used to try to gain access to determine what software the attacker used to start the attack.


# In[2]:


pip install numpy


# In[4]:


pip install pandas


# In[11]:


pip install matplotlib


# In[6]:


import numpy as np
import pandas as pd

df = pd.read_csv("cybersecurity.csv")

df_clean = df.drop(columns=["url", "user_agent"])

print("Missing values per column:\n", df_clean.isnull().sum())

duplicates_count = df_clean.duplicated().sum()
print(f"Duplicate rows identified: {duplicates_count}")
df_clean = df_clean.drop_duplicates()

df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
df_clean["protocol"] = df_clean["protocol"].astype("category")
df_clean["attack_type"] = df_clean["attack_type"].astype("category")
df_clean["is_internal_traffic"] = df_clean["is_internal_traffic"].astype(bool)

df_clean["protocol"] = df_clean["protocol"].str.upper().str.strip()
df_clean["attack_type"] = df_clean["attack_type"].str.lower().str.strip()

q_sent = df_clean["bytes_sent"].quantile(0.99)
q_recv = df_clean["bytes_received"].quantile(0.99)

df_clean["bytes_sent_clean"] = np.where(
    df_clean["bytes_sent"] > q_sent, q_sent, df_clean["bytes_sent"]
)
df_clean["bytes_received_clean"] = np.where(
    df_clean["bytes_received"] > q_recv, q_recv, df_clean["bytes_received"]
)


# In[8]:


#How do network traffic characteristics, communication protocols, and perimeter boundaries differentiate malicious cyber attacks from baseline internal traffic?
#Which communication protocols are primarily leveraged by different methods of cyberattacks compared to normal operations?
#How do the types and patterns of threats differ between inbound external connections and internal network traffic?
#How do data transmission loads vary between different malicious attack categories compared to benign network traffic?


# In[12]:


import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("cybersecurity.csv")
df_clean = df.copy()
df_clean["protocol"] = df_clean["protocol"].str.upper().str.strip()
df_clean["attack_type"] = df_clean["attack_type"].str.lower().str.strip()

malicious_df = df_clean[df_clean["attack_type"] != "benign"]

proto_counts = pd.crosstab(
    malicious_df["attack_type"], malicious_df["protocol"]
)

proto_counts["Total"] = proto_counts.sum(axis=1)
proto_counts = proto_counts.sort_values(by="Total", ascending=True).drop(
    columns=["Total"]
)

plt.figure(figsize=(12, 7))
ax = proto_counts.plot(
    kind="barh",
    figsize=(12, 7),
    color=["#3498db", "#e74c3c", "#2ecc71"],
    width=0.8,
)

plt.title(
    "Sub-Question A: Malicious Attack Frequency by Network Protocol",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Number of Incidents", fontsize=12, fontweight="bold")
plt.ylabel("Attack Type", fontsize=12, fontweight="bold")
plt.legend(title="Protocol", title_fontsize="11", fontsize="10")
plt.grid(axis="x", linestyle="--", alpha=0.5)

for container in ax.containers:
    labels = [int(v) if v > 0 else "" for v in container.datavalues]
    ax.bar_label(container, labels=labels, padding=3, fontsize=9, fontweight="bold")

plt.tight_layout()

plt.savefig("sub_question_a_bar_graph.png")
plt.show()


# In[ ]:


#Interpretation & Findings for Question A
#TCP is the primary protocol utilized across almost all attack vectors, representing over 75% of total recorded security incidents. This is expected since -
# Stateful connection-oriented attacks (such as HTTP-based SQL injections, XSS, and SSH brute-forcing) require reliable two-way packet delivery.

#UDP accounts for notable occurrences in brute-force attacks (20 incidents) and port scans (16 incidents), leveraging UDP's connectionless nature for fast scanning and spoofing.

#ICMP records minimal activity, appearing primarily in reconnaissance scans and select flood signatures.


# In[16]:


import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("cybersecurity.csv")
df_clean = df.copy()
df_clean["attack_type"] = df_clean["attack_type"].str.lower().str.strip()
df_clean["is_internal_traffic"] = df_clean["is_internal_traffic"].astype(bool)

malicious_df = df_clean[df_clean["attack_type"] != "benign"]

pivot_q_b = pd.pivot_table(
    malicious_df,
    index="attack_type",
    columns="is_internal_traffic",
    values="label",
    aggfunc="count",
    fill_value=0,
)

pivot_q_b.columns = ["External Traffic", "Internal Traffic"]

pivot_q_b["Total"] = pivot_q_b.sum(axis=1)
pivot_q_b = pivot_q_b.sort_values(by="Total", ascending=True).drop(
    columns=["Total"]
)

plt.figure(figsize=(10, 6))
ax = pivot_q_b.plot(
    kind="barh", figsize=(10, 6), color=["#2b5c8f", "#d95f02"], width=0.7
)
plt.title(
    "Sub-Question B: Malicious Attacks by Traffic Origin",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Number of Attack Incidents", fontsize=11, fontweight="bold")
plt.ylabel("Attack Type", fontsize=11, fontweight="bold")
plt.legend(title="", fontsize=10)
plt.grid(axis="x", linestyle="--", alpha=0.5)

for container in ax.containers:
    labels = [int(v) if v > 0 else "" for v in container.datavalues]
    ax.bar_label(container, labels=labels, padding=3, fontsize=9, fontweight="bold")

plt.tight_layout()

plt.savefig("sub_question_b_simplified.png")
plt.show()


# In[ ]:


#Interpretation & Findings for Question B
#Over 84% of all malicious events originate from external networks. Command Injection (92.3%) and Credential Stuffing (89.3%) exhibit the highest external concentration.
#Internal traffic is not immune to security incidents. Roughly 15% of brute-force and port-scanning events occur within internal subnets, highlighting lateral movement.
# Scenarios where compromised internal endpoints scan adjacent subnet hosts.


# In[20]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("cybersecurity.csv")
df_clean = df.copy()
df_clean["attack_type"] = df_clean["attack_type"].str.lower().str.strip()

q_sent = df_clean["bytes_sent"].quantile(0.99)
q_recv = df_clean["bytes_received"].quantile(0.99)

df_clean["bytes_sent_clean"] = np.where(
    df_clean["bytes_sent"] > q_sent, q_sent, df_clean["bytes_sent"]
)
df_clean["bytes_received_clean"] = np.where(
    df_clean["bytes_received"] > q_recv, q_recv, df_clean["bytes_received"]
)

df_clean["total_bytes_clean"] = (
    df_clean["bytes_sent_clean"] + df_clean["bytes_received_clean"]
)

malicious_df = df_clean[df_clean["attack_type"] != "benign"]

payload_by_attack = (
    malicious_df.groupby("attack_type")["total_bytes_clean"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 9))
colors = plt.cm.tab10(np.linspace(0, 1, len(payload_by_attack)))

plt.pie(
    payload_by_attack,
    labels=payload_by_attack.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    textprops={"fontsize": 11, "weight": "bold"},
)

plt.title(
    "Sub-Question C: Proportion of Total Data Transfer Volume (Bytes) by Malicious Attack Category",
    fontsize=13,
    fontweight="bold",
    pad=20,
)

plt.tight_layout()

plt.savefig("sub_question_c_pie_chart.png")
plt.show()


# In[ ]:


#Interpretation & Findings for Question C
#DDoS attacks produce massive asymmetric bandwidth anomalies, averaging ~831 KB sent and ~1.62 MB received per logged session—roughly 10x the bandwidth usage of normal baseline traffic.

#Port scanning displays the lowest data payload (6.5 KB sent / 15.7 KB received), confirming that reconnaissance attacks intentionally minimize packet payload sizes to evade traditional volume-threshold intrusion detection systems (IDS).

#C2 beaconing shows elevated data exfiltration figures (111 KB sent / 184 KB received), reflecting persistent two-way heartbeat and payload communication.


# In[21]:


#Project Findings
#TCP is the dominant protocol for enterprise network cyber attacks (over 75%), making TCP-state inspection and application-layer firewalling essential.

#External traffic is the primary intrusion vector (84%), but internal subnet monitoring remains critical to stop lateral brute-force propagation.

#Network traffic payload volume is a key indicator for threat classification: DDoS causes extreme volumetric spikes, while port scans remain stealthy with minimal data footprints.


# In[ ]:


#Unanswered Questions
# Due to privacy and limited logging, raw packet payloads for deep packet inspection were not available.

#The dataset lacked sequential session IDs, preventing the tracking of multi-stage kill chains

