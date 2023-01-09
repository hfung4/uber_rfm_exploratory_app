# data
import streamlit as st
import pandas as pd
import numpy as np
import math
import statsmodels
import statsmodels.api as sm
import s3fs

# common
from pathlib import Path

# visualization
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# custom modules
import data_manager as dm
import processing as pp

st.title("Exploratory analyses on the effects of Rides for Moms on prenatal attendance")
st.subheader("A Web App by [Henry Fung](https://github.com/hfung4)")
st.write(
    "Using attendance, uber rides and demographics data, we investigate the effects of the Rides for Mom program on prenatal in-person visits"
)

# add linespace
st.text("")
st.text("")
st.text("")

# Import config -------------------------------------------
config = dm.load_yaml("config.yaml")


# Import data ----------------------------------------------

# Import data from S3
control_raw = dm.load_data_s3(bucket_name=config["BUCKET_NAME"], key_name="control.csv")
rides_raw = dm.load_data_s3(bucket_name=config["BUCKET_NAME"], key_name="rides.csv")
onsite_att_raw = dm.load_data_s3(
    bucket_name=config["BUCKET_NAME"], key_name="onsite_att.csv"
)

# If load data from flat file (testing only)
# control_raw = dm.load_data(Path("data", "input_data"), "control.csv")
# rides_raw = dm.load_data(Path("data", "input_data"), "rides.csv")
# onsite_att_raw = dm.load_data(Path("data", "input_data"), "onsite_att.csv")


# Data processing ----------------------------------------

onsite_att = onsite_att_raw.copy(deep=True)  # make a copy of onsite_att
control = control_raw.copy(deep=True)
rides = rides_raw.copy(deep=True)

# Mapping dictionary that maps var_label to var_name
dict_mapping = config["DICT_MAPPING"]

# Remove the second pregnancy of patient 1800053724 to remove duplicated Patient.ID (for now, probably need another ID)
control = control.loc[
    ~((control["Patient.ID"] == 1800053724) & (control["Inactive.Reason"].isnull())), :
]


# Create flags for patients with no EDD or Delivery Date information,
# excluded appointment status, virtual visits, other excluded visit types,
# and post partum appts
onsite_att["missing_edd_or_delivery"] = np.where(
    ((onsite_att["EDD_Date"].isnull()) & (onsite_att["Delivery_Date"].isnull())),
    True,
    False,
)
onsite_att["status_type_excl"] = np.where(
    onsite_att["STATUS"].isin(config["EXCL_STATUS_TYPES"]), True, False
)
onsite_att["virtual_visits"] = np.where(
    onsite_att["Appointment_Type"].isin(config["VIRTUAL_VISIT_TYPES"]), True, False
)
onsite_att["other_visits_excl"] = np.where(
    onsite_att["Appointment_Type"].isin(config["OTHER_APPT_TYPES_EXCLUDED"]),
    True,
    False,
)
onsite_att["post_partum"] = np.where(
    onsite_att["Appointment_Type"].isin(config["POST_PARTUM_APPT"]),
    True,
    False,
)

# A flag for all appointments with appointment dates later than the EDD_Date or the Delivery Date, and it is not a
# post-partum appointment.
onsite_att["post_delivery_non_pp"] = np.where(
    (
        (
            (onsite_att["Appointment_Date"] > onsite_att["EDD_Date"])
            | (onsite_att["Appointment_Date"] > onsite_att["Delivery_Date"])
        )
        & ~onsite_att["post_partum"]
    ),
    True,
    False,
)

# Output the count and percentage of each excluded groups
def get_count_percent(df, var, label):
    count = df[var].value_counts().filter(items=[True], axis="index").values[0]
    percentage = (
        100
        * df[var]
        .value_counts(normalize=True)
        .filter(items=[True], axis="index")
        .values[0]
    )
    return pd.DataFrame(
        {"count": count, "percentage": math.ceil(percentage)}, index=[label]
    )


df_cnt_percent = pd.DataFrame()
for var, label in [
    ("missing_edd_or_delivery", "missing EDD or Delivery Date"),
    (
        "status_type_excl",
        "excluded appointment status (cancelled by HC, or rescheduled)",
    ),
    ("virtual_visits", "virtual appointments"),
    ("other_visits_excl", "clinic staffs visit the homes of patients"),
    ("post_partum", "post partum appointments"),
    ("post_delivery_non_pp", "post delivery appts, not post-partum"),
]:
    df_cnt_percent = pd.concat(
        [df_cnt_percent, get_count_percent(onsite_att, var, label)]
    )


# Filter out appointments that are in the excluded status type, virtual visits, other visit types that are excluded
# and post-partum visits
onsite_subset = onsite_att.loc[
    (
        ~onsite_att["missing_edd_or_delivery"]
        & ~onsite_att["status_type_excl"]
        & ~onsite_att["virtual_visits"]
        & ~onsite_att["other_visits_excl"]
        & ~onsite_att["post_partum"]
        & ~onsite_att["post_delivery_non_pp"]
    ),
    :,
]

st.header("Data Processing")
st.subheader(
    f"In total, {onsite_subset.shape[0]} appointments entered into our analyses"
)

st.markdown("**Below is a breakdown of the excluded appointments**")

# Output table
st.table(df_cnt_percent.reset_index())

st.markdown(
    "**In summary, we excluded all appointments whose patients have missing EDD or Delivery Date, virtual appointments (including group/centering appointments in MC), in-person appointments that were cancelled by the HC or were rescheduled, appointments in which clinic staffs visited the home of the patient, post-partum appointments, and non post-partum appointments that are later than the reported delivery date.**"
)

# add linespace
st.text("")
st.text("")
st.text("")


# Comparision of completion percentage between Community of Hope and Mary Center ---------------------------

# Data processing

# Left join onsite_subset with control by Patient.ID
onsite_control = pd.merge(onsite_subset, control, on="Patient.ID", how="left")

onsite_control["Appointment status"] = np.where(
    ((onsite_control["STATUS"] == "completed") | (onsite_control["STATUS"] == "CHK")),
    "Completed",
    "Not completed",
)

# Create a variable called int_enrolled that takes on 1 for all appointments of patients enrolled in RfM and are not declined
onsite_control["Enrollment status"] = np.where(
    (~onsite_control["Enrollment.Date"].isnull())
    & (onsite_control["Status"] == "Participants"),
    "Enrolled",
    "Not enrolled/declined",
)

# Rename "Health.Center_x" to Health Center
onsite_control = onsite_control.rename(columns={"Health.Center_x": "Health.Center"})


# Get frequency table
df_plot = (
    onsite_control.groupby("Health.Center")["Appointment status"]
    .value_counts(normalize=True)
    .rename("percentage")
    .multiply(100)
    .reset_index()
)

df_plot["percentage"] = np.round(df_plot["percentage"], 1)


# create stacked bar chart
fig = px.bar(
    df_plot,
    y="Health.Center",
    x="percentage",
    orientation="h",
    color="Appointment status",
    title="Appointment completion comparision between health centers",
    labels={"Health.Center": "Health Center"},
    text_auto=True,
)

fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
    font={"size": 18},
)


st.subheader(
    """1. Community of Hope has a lower completion percentage of in-person prenatal visits (81.41%) compared to Mary Center (92.17%)"""
)

st.plotly_chart(fig)

# add linespace
st.text("")
st.text("")
st.text("")


# Comparision of RfM enrollment between Community of Hope and Mary Center -------------------------------------


# Plot the percentage enrolled/not enrolled appointments for each health center

# Get frequency table
df_plot = (
    onsite_control.groupby("Health.Center")["Enrollment status"]
    .value_counts(normalize=True)
    .rename("percentage")
    .multiply(100)
    .reset_index()
)

# Round percentage
df_plot["percentage"] = np.round(df_plot["percentage"], 1)

# create stacked bar chart
fig = px.bar(
    df_plot,
    y="Health.Center",
    x="percentage",
    orientation="h",
    color="Enrollment status",
    color_discrete_sequence=px.colors.qualitative.Antique,
    title="Enrollment status comparision between health centers",
    labels={"Health.Center": "Health Center"},
    text_auto=True,
)

fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
    font={"size": 18},
)


st.subheader(
    """2. Amongst the in-person prenatal appointments in Community of Hope, 35% of them were made by patients who were enrolled in RfM. This is substantially higher than the 13% made by Mary Center patients who were enrolled in RfM."""
)

st.plotly_chart(fig)

# add linespace
st.text("")
st.text("")
st.text("")


# Percentage of completed appointments made by patients who are enrolled and patients who are not enrolled ---------------

st.subheader(
    """3. The percentage of completed appointments are similar between those made by patients enrolled in RfM and those who did not enrolled in RfM. This holds true whether I group by Health Center or not."""
)

st.text("")

# Select box
values = ["Yes", "No"]
default_ix = values.index("No")
group_by_hc = st.selectbox(
    "Do you want to group by Health Center?", values, index=default_ix
)

if group_by_hc == "Yes":
    df_plot = (
        onsite_control.groupby(["Enrollment status", "Health.Center"])[
            "Appointment status"
        ]
        .value_counts(normalize=True)
        .rename("percentage")
        .multiply(100)
        .reset_index()
    )
    facet_var = "Health.Center"
else:
    df_plot = (
        onsite_control.groupby("Enrollment status")["Appointment status"]
        .value_counts(normalize=True)
        .rename("percentage")
        .multiply(100)
        .reset_index()
    )
    facet_var = None

df_plot["percentage"] = np.round(df_plot["percentage"], 1)


# Stack plots of completion percentage of COH and MC appointments
fig = px.bar(
    df_plot,
    y="Enrollment status",
    x="percentage",
    orientation="h",
    facet_row=facet_var,
    color="Appointment status",
    color_discrete_sequence=px.colors.qualitative.D3,
    title="Appointment status comparision between the enrolled and not enrolled",
    text_auto=True,
)

fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
    font={"size": 18},
)


st.plotly_chart(fig)

# add linespace
st.text("")
st.text("")
st.text("")


# The distribution of completed in-person onsite visits per patient by Health Center ---------------------------------


# Get the number of completed visits per patient for each HC

# Create a new variable called Appointment completed that takes on 1 if Appointment status is completed, and 0 otherwise
onsite_control["Appointment completed"] = np.where(
    onsite_control["Appointment status"] == "Completed", 1, 0
)

# Compute the number of completed appointments and completion percentage of each patient in each HC
df_plot = (
    onsite_control.groupby(["Patient.ID", "Health.Center"])
    .agg(
        n_completes=("Appointment completed", "sum"),
        p_completes=("Appointment completed", lambda x: round(100 * np.mean(x), 2)),
    )
    .reset_index()
)

# Get the total number of unique patients
n_patients = df_plot["Patient.ID"].nunique()
n_patients_mc = df_plot.loc[df_plot["Health.Center"] == "MC", "Patient.ID"].nunique()
n_patients_coh = df_plot.loc[df_plot["Health.Center"] == "COH", "Patient.ID"].nunique()


st.subheader(
    "4. There are on average 5.22 completed in-person onsite visits per patient in MC, and 6.18 for COH. Both distributions are right-skewed."
)
st.subheader(
    "Around 70% of patients in MC completed all of their booked appointments (completion percentage of 100), and 40% of COH patients."
)

# NOTE: the average of averages is not equal to the average of al the numbers originally averaged. As an example, the average of the
# completion percentages of all patients in MC (an average of averages) is slightly different from the completion percentage
# of appointments in MC (an average of all the numbers originally averaged).
# I comment the below for now to avoid confusion.

# st.subheader(
#    "On average, MC patients completed 91% of their booked appointments and COH patients completed 82% of their booked appointments. A #large number of patients completed all of their booked appointments."
# )


df_unique_patients = pd.DataFrame(
    {
        "Health Centers": ["Mary Center", "Community of Hope", "Total"],
        "Number of patients": [n_patients_mc, n_patients_coh, n_patients],
    }
)

# Hide index in table
# CSS to inject contained in a string
hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

st.text("")
st.markdown("**Number of unique patients**")
st.table(df_unique_patients)


# Summary statistics of completed in person appointments of patients by Health.Center.
df_summary_cnt = (
    df_plot.groupby("Health.Center")
    .agg(
        mean_num_of_completes=("n_completes", "mean"),
        max_num_of_completes=("n_completes", "max"),
        min_num_of_completes=("n_completes", "min"),
    )
    .reset_index()
)


# Summary statistics of completion percentage by Health.Center.
df_summary_pct = (
    df_plot.groupby("Health.Center")
    .agg(
        mean_pct_of_completes=("p_completes", "mean"),
        max_pct_of_completes=("p_completes", "max"),
        min_pct_of_completes=("p_completes", "min"),
    )
    .reset_index()
)


# Number of patients who completed 100% of their booked appointments

n_patients_100 = df_plot.loc[df_plot["p_completes"] == 100, "p_completes"].count()
n_patients_mc_100 = df_plot.loc[
    (df_plot["p_completes"] == 100) & (df_plot["Health.Center"] == "MC"), "p_completes"
].count()
n_patients_coh_100 = df_plot.loc[
    (df_plot["p_completes"] == 100) & (df_plot["Health.Center"] == "COH"), "p_completes"
].count()


df_summary_100 = pd.DataFrame(
    {
        "Health Centers": ["Mary Center", "Community of Hope", "Overall"],
        "Percentage": [
            round(100 * n_patients_mc_100 / n_patients_mc, 2),
            round(100 * n_patients_coh_100 / n_patients_coh, 2),
            round(100 * n_patients_100 / n_patients, 2),
        ],
    }
)

# Let user select whether we display information about number of completed appointments or completion percentage

st.text("")
st.text("")
st.text("")

# Select box
values = [
    "Number of completed appointments",
    "Appointments completion percentage",
]
default_ix = values.index("Number of completed appointments")
completed_appt_mode = st.selectbox(
    "Select completion count or percentage", values, index=default_ix
)


# Count
if completed_appt_mode == "Number of completed appointments":
    var = "n_completes"
    x_label = {"n_completes": "Number of completed appointment per patient"}

    # Write df_summary_cnt to app
    st.text("")
    st.text("")
    st.markdown("**Summary statistics of number of completed appointments**")
    st.table(df_summary_cnt)

else:
    # Percentage
    var = "p_completes"
    x_label = {"p_completes": "Appointments completion percentage per patient"}

    # Write df_summary_pct to app
    st.text("")
    st.text("")
    st.markdown("**Summary statistics of completed percentage**")
    # st.table(df_summary_pct) Comment out for now-- average of averages vs average of all numbers originally averaged.

    # Write df_summary_100 to app
    st.text("")
    st.markdown(
        "**Percentage of patients in each Health Center who completed all booked appointments**"
    )
    st.table(df_summary_100)


fig = px.histogram(
    df_plot,
    x=var,
    color="Health.Center",
    barmode="overlay",
    marginal="box",
    labels=x_label,
)

fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
    font={"size": 18},
)

st.plotly_chart(fig)


# Examine percentage of patients in MC who exceeds 6 in person prenatal visits and the outlier with 27 visits --------------------------


mc_n_completes_gt_6 = df_plot.loc[
    (df_plot["Health.Center"] == "MC") & (df_plot["n_completes"] > 6), :
].shape[0]

mc_p_completes_gt_6 = round(100 * mc_n_completes_gt_6 / n_patients_mc, 2)


df_mc_gt_6 = pd.DataFrame(
    {
        "Number of MC patients": [mc_n_completes_gt_6],
        "Percentage of MC patients": [mc_p_completes_gt_6],
    }
)


st.text("")
st.text("")
st.text("")
st.subheader(
    "5. It is known that the recommended number of in-person prenatal appointments of Mary Center is 6. Let's examine the percentage of patients exceeding this threshold and an outlier patient from Mary Center with 27 appointments"
)

st.text("")
st.markdown(
    "**The number and percentage of Mary Center patients who have more than 6 in-person appointments**"
)
st.table(df_mc_gt_6)

# Get the patient ID of the patient who had 27 in person visits
patient_id_27 = df_plot.loc[
    (df_plot["Health.Center"] == "MC") & (df_plot["n_completes"] == 27), "Patient.ID"
].values[0]

st.markdown("**Let's look at the MC patient who had 27 in-person visits**")

st.text("")
show_table = st.radio("Show table", ["Don't Show", "Show"], 0)

if show_table == "Show":
    # Get the patient ID of the patient who had 27 in person visits
    patient_id_27 = df_plot.loc[
        (df_plot["Health.Center"] == "MC") & (df_plot["n_completes"] == 27),
        "Patient.ID",
    ].values[0]
    # Get information about this patient
    df_patient_mc_27 = onsite_control.loc[
        onsite_control["Patient.ID"] == patient_id_27, :
    ].sort_values("Appointment_Date")
    st.table(df_patient_mc_27)

st.markdown(
    "**In summary, 31.86% of the patients in Mary Center have more than 6 in-person onsite visits, which exceeds the recommended number of appointments according to the MC scehdule. The outlier patient did not enroll in RfM. She completed all her booked appointments, but 15 of her 27 visits have 'NURSE' as the appointment type (this type did not appear in codebook). Further filtering of appointment type might be needed.**"
)


# Number and percentage of completed visits between the enrolled and the not enrolled ----------------------------------

st.text("")
st.text("")
st.text("")
st.subheader(
    "6. On average, the number of completed in-person prenatal visits per patient is slighly higher for patients enrolled in RfM (6.87 visits per patient) and not enrolled (5.18 visits per patient)."
)

st.subheader(
    "As seen previously (Result 3), the completion percentage between the enrolled and the not-enrolled is similar. Here, we look at the distribution of the per-patient completion percentage as well, and we see that a large number of enrolled/not-enrolled patients completed all of her booked appointment."
)

# Get the number of completed visits per patient for each HC

# Compute the number of completed appointments and completion percentage of each patient who are in the enrolled and not enrolled group
df_plot = (
    onsite_control.groupby(["Patient.ID", "Enrollment status"])
    .agg(
        n_completes=("Appointment completed", "sum"),
        p_completes=("Appointment completed", lambda x: round(100 * np.mean(x), 2)),
    )
    .reset_index()
)

# Summary statistics of completed in person appointments of patients by Enrollment Status
df_summary_cnt_enroll = (
    df_plot.groupby("Enrollment status")
    .agg(
        mean_num_of_completes=("n_completes", "mean"),
        max_num_of_completes=("n_completes", "max"),
        min_num_of_completes=("n_completes", "min"),
    )
    .reset_index()
)


# Summary statistics of completion percentage by Enrollment Status.
# We won't display this result due to the confusion of "the average of averages is not equal to the average of al the numbers originally averaged".
df_summary_pct_enroll = (
    df_plot.groupby("Enrollment status")
    .agg(
        mean_pct_of_completes=("p_completes", "mean"),
        max_pct_of_completes=("p_completes", "max"),
        min_pct_of_completes=("p_completes", "min"),
    )
    .reset_index()
)

# Get the total number of unique patients
n_patients = df_plot["Patient.ID"].nunique()
n_patients_enrolled = df_plot.loc[
    df_plot["Enrollment status"] == "Enrolled", "Patient.ID"
].nunique()
n_patients_nenrolled = df_plot.loc[
    df_plot["Enrollment status"] == "Not enrolled/declined", "Patient.ID"
].nunique()

# Number of enrolled and not enrolled patients
df_unique_patients_enroll = pd.DataFrame(
    {
        "Enrollment status": ["Enrolled", "Not enrolled/declined", "Total"],
        "Number of patients": [n_patients_enrolled, n_patients_nenrolled, n_patients],
    }
)

st.text("")
st.table(df_unique_patients_enroll)


# Let user select whether we display information about number of completed appointments or completion percentage

st.text("")
st.text("")
st.text("")

# Select box
values = [
    "Number of completed appointments",
    "Appointments completion percentage",
]
default_ix = values.index("Number of completed appointments")
completed_appt_mode = st.selectbox(
    "Select completion count or percentage", values, index=default_ix, key=1
)


# Count
if completed_appt_mode == "Number of completed appointments":
    var = "n_completes"
    x_label = {"n_completes": "Number of completed appointment per patient"}

    # Write df_summary_cnt to app
    st.text("")
    st.text("")
    st.markdown("**Summary statistics of number of completed appointments**")
    st.table(df_summary_cnt_enroll)

else:
    # Percentage
    var = "p_completes"
    x_label = {"p_completes": "Appointments completion percentage per patient"}

    # I commented this out because of the confusion this might cause when comparing to result # 3:
    # "the average of averages is not equal to the average of al the numbers originally averaged"
    # Write df_summary_pct to app
    # st.text("")
    # st.text("")
    # st.markdown("**Summary statistics of completed percentage**")
    # st.table(df_summary_pct_enroll)


fig = px.histogram(
    df_plot,
    x=var,
    color="Enrollment status",
    barmode="overlay",
    marginal="box",
    labels=x_label,
)

fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
    font={"size": 18},
)

st.plotly_chart(fig)


## Relationship between weeks before delivery and percentage of completed appointments ------------------------------------

st.text("")
st.text("")
st.text("")

st.subheader(
    "7. There is a positive association between weeks before delivery and percentage of completed appointments."
)

st.subheader(
    "Interpretation: on average, as we get closer to the delivery date, the percentage of completed appointments decreases. But overall, the (average) completion percentage are high for all weeks (>85%)"
)

st.subheader(
    "Breaking down by enrollment status, for patients who are enrolled in RfM, their in-persons appointment completion percentage increases as their delivery date gets closer. In contrast, there is a decrease in completion percentage of patients who are not enrolled. That said, the completion percentage of the enrolled starts off lower (by about 10%) compared to patients who are not enrolled."
)

# Data processing

# Filter out all appointments whose patient has missing Delivery Date
df_delivery = onsite_control.loc[~onsite_control["Delivery_Date"].isnull()]
# Invert the sign of delta_from_delivery
df_delivery["delta_from_delivery"] = -df_delivery["delta_from_delivery"]


# Assuming a woman participate in the program starting from the first days of pregnancy,
# at most, her appointment should be only 280 days (40 weeks) from delivery date.
# Filter out appointments whose delta from delivery is more than 280 days
df_delivery = df_delivery.loc[df_delivery["delta_from_delivery"] <= 280, :]

# Bin days from delivery into 7 day periods to get weeks from delivery
df_delivery["weeks_from_delivery"] = pd.cut(
    df_delivery["delta_from_delivery"],
    bins=np.arange(start=0, stop=287, step=7).tolist(),
    labels=np.arange(start=1, stop=41, step=1).tolist(),
    include_lowest=True,
)
# Convert to numeric column
df_delivery["weeks_from_delivery"] = df_delivery["weeks_from_delivery"].astype(int)


# Select box
st.text("")

values = ["Yes", "No"]
default_ix = values.index("No")
group_by_enrolled = st.selectbox(
    "Do you want to group by Enrollment status?", values, index=default_ix
)

if group_by_enrolled == "Yes":
    df_plot = (
        df_delivery.groupby(["weeks_from_delivery", "Enrollment status"])
        .agg(
            n_completes=("Appointment completed", "sum"),
            p_completes=("Appointment completed", lambda x: round(100 * np.mean(x), 2)),
        )
        .reset_index()
    )
    facet_var = "Enrollment status"

else:
    # Compute the completion percentage for each week from the delivery date
    df_plot = (
        df_delivery.groupby("weeks_from_delivery")
        .agg(
            n_completes=("Appointment completed", "sum"),
            p_completes=("Appointment completed", lambda x: round(100 * np.mean(x), 2)),
        )
        .reset_index()
    )
    facet_var = None

# Plot the average completion percentage vs week before delivery
fig = px.scatter(
    df_plot,
    x="weeks_from_delivery",
    y="p_completes",
    trendline="ols",
    labels={
        "weeks_from_delivery": "Weeks from Reported Delivery Date",
        "p_completes": "Average completion percentage",
    },
    facet_col=facet_var,
)

fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
    font={"size": 18},
)
st.plotly_chart(fig)


# Regression model: regression completion percentage on rides per appointment ------------------------------------------

st.text("")
st.text("")
st.text("")

st.subheader(
    "8. Modelling: association between rides per appointment and completion percentage of booked appointments"
)

# Data processing

# Create a dataframe that contains the number of completed appointments and completion percentage of each patient
# in each Health.Center
df_appt = (
    onsite_control.groupby(["Patient.ID", "Health.Center"])
    .agg(
        n_appt=("Appointment completed", "count"),
        n_appt_completes=("Appointment completed", "sum"),
        p_appt_completes=(
            "Appointment completed",
            lambda x: round(100 * np.mean(x), 2),
        ),
    )
    .reset_index()
)

# Create a dataframe that contains the number of completed rides, average duration, and average distance per patient

# First, create a new variable called "ride_completed" that takes on 1 if the ride is completed, and 0 otheriwse
rides["rides_completed"] = np.where(rides["status"] == "completed", 1, 0)

# I also want to rename "status" to "ride_status"
rides = rides.rename(columns={"status": "ride_status"})

# Create a dataframe of aggregated rows by Patient
df_rides = rides.groupby(["Patient.ID"]).agg(
    n_rides_completes=("rides_completed", "sum"),
    mean_distance_travelled=("trip.distance.miles", "mean"),
    mean_ride_duration=("trip.duration.minutes", "mean"),
)

# Inner join df_appt and df_rides so that I will only include patients of whom we have ride data (took at least one ride)
# All patients who completed at least one ride are enrolled in RfM. I verified that those who are not enrolled in RfM
# did not complete a single ride


df_appt_rides = pd.merge(df_appt, df_rides, how="inner", on="Patient.ID")

# Inner join df_appt_rides with control so we have access to demographic information of patients
# There are 244 patients in the dataframe
onsite_control_rides = pd.merge(df_appt_rides, control, how="inner", on="Patient.ID")

# Rename Health.Center_x variable
onsite_control_rides = onsite_control_rides.rename(
    columns={"Health.Center_x": "Health.Center"}
)

# Compute the rides_per_appt feature
onsite_control_rides["rides_per_appt"] = (
    onsite_control_rides["n_rides_completes"] / onsite_control_rides["n_appt"]
)

# Derive a few more features:
# freq_users gets 1 if rides_per_appt >0.5 and 0 Otheriwse
# completed_all_appts gets 1 if p_appt_completes == 100

onsite_control_rides["freq_users"] = np.where(
    onsite_control_rides["rides_per_appt"] > 0.5, 1, 0
)

onsite_control_rides["completed_all_appts"] = np.where(
    onsite_control_rides["p_appt_completes"] == 100, 1, 0
)


# Distribution of features and outcome variable

st.markdown(
    "**Let's look at the distribution of the features and the outcome variables**"
)

# Group by health center option
st.text("")
distrb_group_by_hc = st.radio("Group by Health Center", ["No", "Yes"], 0, key=2)

if distrb_group_by_hc == "Yes":
    group_by_var = "Health.Center"
else:
    group_by_var = None


# Variable selector
st.text("")

# Select box
values = [
    "appointments completion percentage",
    "number of completed appointments",
    "number of completed rides",
    "number of total appointments",
    "rides per appointment",
    "mean distance travelled on Uber (miles)",
    "mean ride duration on Uber (minutes)",
    "completed all appointments",
    "frequent users (rides per appointment ratio >0.5)",
]
default_ix = values.index("appointments completion percentage")
var_label = st.selectbox(
    "Select the variable for the hisrogram", values, index=default_ix
)


# get variable name
var = dict_mapping[var_label]
x_label = {var: var_label}

fig = px.histogram(
    onsite_control_rides,
    x=var,
    color=group_by_var,
    barmode="overlay",
    marginal="box",
    labels=x_label,
)

fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
    font={"size": 18},
)

st.plotly_chart(fig)


# Remove 3 patients for outlier rides_per_appt
onsite_control_rides = pp.remove_outliers_iqr(
    df=onsite_control_rides, vars=["rides_per_appt"], plotting=False
)


# Modelling ----------------------------------------------------------------

st.markdown("### Modelling")

# Choosing IV and DV


with st.form("user_inputs_regr"):
    iv_selected = st.multiselect(
        "Choose your independent variables: ", config["IV_OPTIONS"]
    )
    # write the selected options
    st.write("You selected", len(iv_selected), "independent variables")

    dv_selected = st.selectbox(
        "Choose your dependent variables: ", config["DV_OPTIONS"]
    )

    st.form_submit_button()

# Populate IV and DV by mapping var labels to var names
IV = [dict_mapping[var_label] for var_label in iv_selected]
DV = dict_mapping[dv_selected]

# IV and DV
X_train = onsite_control_rides[IV]
X_train = sm.add_constant(X_train)

y_train = onsite_control_rides[DV]


# OLS regression
# Regress p_appt_completes on rides_per_appt

if DV == "p_appt_completes":

    # Fit model
    reg = sm.OLS(y_train, X_train).fit()

    # Results
    reg_results = reg.summary()

    st.text("")
    st.text("")
    st.text("")

    st.markdown("### OLS regression outputs")
    st.write(reg_results)

elif DV == "completed_all_appts":

    # Logreg
    # Regress completed_all_appts on rides_per_appt

    # Fit model
    clf = log_reg = sm.Logit(y_train, X_train).fit()

    # Results
    clf_results = clf.summary()

    st.markdown("### Logisitic regression outputs")
    st.write(clf_results)
    st.text("")
    st.text("")
    st.markdown("#### Coefficients in odds-ratio")
    st.write(np.exp(clf.params))
