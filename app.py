import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import datetime

st.set_page_config(page_title="CPM Schedule Program", layout="wide")

def get_ef(es, duration, weekmask, holidays):
    if duration <= 0:
        return np.busday_offset(es, 0, roll='forward', weekmask=weekmask, holidays=holidays).astype('M8[D]').astype(datetime.date)
    return np.busday_offset(es, duration - 1, roll='forward', weekmask=weekmask, holidays=holidays).astype('M8[D]').astype(datetime.date)

def get_next_es(ef, weekmask, holidays):
    return np.busday_offset(ef, 1, roll='forward', weekmask=weekmask, holidays=holidays).astype('M8[D]').astype(datetime.date)

def get_ls(lf, duration, weekmask, holidays):
    if duration <= 0:
        return np.busday_offset(lf, 0, roll='backward', weekmask=weekmask, holidays=holidays).astype('M8[D]').astype(datetime.date)
    return np.busday_offset(lf, -(duration - 1), roll='backward', weekmask=weekmask, holidays=holidays).astype('M8[D]').astype(datetime.date)

def get_prev_lf(ls, weekmask, holidays):
    return np.busday_offset(ls, -1, roll='backward', weekmask=weekmask, holidays=holidays).astype('M8[D]').astype(datetime.date)

def get_tf(es, ls, weekmask, holidays):
    return int(np.busday_count(es, ls, weekmask=weekmask, holidays=holidays))

def calculate_cpm(df, start_date, weekmask, holidays):
    # Clean DataFrame
    df = df.dropna(subset=['Activity Name']).copy()
    df['Activity Name'] = df['Activity Name'].astype(str).str.strip()
    
    # Create directed graph
    G = nx.DiGraph()
    for index, row in df.iterrows():
        act = row['Activity Name']
        try:
            dur = int(row['Duration'])
        except (ValueError, TypeError):
            dur = 0
        G.add_node(act, duration=dur)
        
    for index, row in df.iterrows():
        act = row['Activity Name']
        preds_str = str(row.get('Predecessors', ''))
        if preds_str.strip() and preds_str.lower() != 'nan':
            preds = preds_str.split(',')
            for p in preds:
                p = p.strip()
                if p:
                    if p not in G.nodes:
                        st.error(f"Predecessor '{p}' for activity '{act}' not found in activity list.")
                        return None
                    G.add_edge(p, act)
                
    if not nx.is_directed_acyclic_graph(G):
        st.error("Cycle detected in predecessors! Please check your dependencies for circular logic.")
        return None
        
    topo_order = list(nx.topological_sort(G))
    
    # Forward Pass
    es_dict = {}
    ef_dict = {}
    
    for node in topo_order:
        duration = G.nodes[node]['duration']
        preds = list(G.predecessors(node))
        
        if not preds:
            # Must start on a working day
            es = np.busday_offset(start_date, 0, roll='forward', weekmask=weekmask, holidays=holidays).astype('M8[D]').astype(datetime.date)
        else:
            max_ef = max([ef_dict[p] for p in preds])
            es = get_next_es(max_ef, weekmask, holidays)
            
        ef = get_ef(es, duration, weekmask, holidays)
        
        es_dict[node] = es
        ef_dict[node] = ef
        
    # Backward Pass
    ls_dict = {}
    lf_dict = {}
    
    if ef_dict:
        project_end_date = max(ef_dict.values())
    else:
        project_end_date = np.busday_offset(start_date, 0, roll='forward', weekmask=weekmask, holidays=holidays).astype('M8[D]').astype(datetime.date)
    
    for node in reversed(topo_order):
        duration = G.nodes[node]['duration']
        succs = list(G.successors(node))
        
        if not succs:
            lf = project_end_date
        else:
            min_ls = min([ls_dict[s] for s in succs])
            lf = get_prev_lf(min_ls, weekmask, holidays)
            
        ls = get_ls(lf, duration, weekmask, holidays)
        
        ls_dict[node] = ls
        lf_dict[node] = lf
        
    # Total Float
    tf_dict = {}
    cp_dict = {}
    
    for node in topo_order:
        tf = get_tf(es_dict[node], ls_dict[node], weekmask, holidays)
        tf_dict[node] = tf
        cp_dict[node] = (tf == 0)
        
    result_df = df.copy()
    result_df['ES'] = result_df['Activity Name'].map(lambda x: es_dict.get(x, None))
    result_df['EF'] = result_df['Activity Name'].map(lambda x: ef_dict.get(x, None))
    result_df['LS'] = result_df['Activity Name'].map(lambda x: ls_dict.get(x, None))
    result_df['LF'] = result_df['Activity Name'].map(lambda x: lf_dict.get(x, None))
    result_df['Total Float'] = result_df['Activity Name'].map(lambda x: tf_dict.get(x, None))
    result_df['Critical Path'] = result_df['Activity Name'].map(lambda x: cp_dict.get(x, False))
    
    return result_df


# --- UI Setup ---
st.title("🚧 CPM & Schedule Program")
st.markdown("A simple web-based Critical Path Method calculator with calendar integration.")

# Sidebar Calendar Settings
st.sidebar.header("📅 Calendar Settings")
start_date = st.sidebar.date_input("Project Start Date", datetime.date.today())

st.sidebar.subheader("Work Days")
col1, col2 = st.sidebar.columns(2)
mon = col1.checkbox("Monday", value=True)
tue = col1.checkbox("Tuesday", value=True)
wed = col1.checkbox("Wednesday", value=True)
thu = col1.checkbox("Thursday", value=True)
fri = col1.checkbox("Friday", value=True)
sat = col2.checkbox("Saturday", value=False)
sun = col2.checkbox("Sunday", value=False)

weekmask_list = [
    '1' if mon else '0',
    '1' if tue else '0',
    '1' if wed else '0',
    '1' if thu else '0',
    '1' if fri else '0',
    '1' if sat else '0',
    '1' if sun else '0',
]
weekmask = "".join(weekmask_list)

st.sidebar.subheader("Holidays (Non-Working Days)")
holidays_input = st.sidebar.text_area("Enter dates (YYYY-MM-DD), one per line:")
holidays = []
if holidays_input:
    for line in holidays_input.split('\n'):
        line = line.strip()
        if line:
            try:
                datetime.datetime.strptime(line, '%Y-%m-%d')
                holidays.append(line)
            except ValueError:
                st.sidebar.error(f"Invalid date format: {line}")

# Main Input Section
st.header("1. Activity Inputs")
st.markdown("Enter your tasks below. Use comma separation for multiple predecessors (e.g., `A, B`). Leave empty if none.")

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({
        'Activity Name': ['A', 'B', 'C', 'D', 'E'],
        'Duration': [3, 4, 2, 5, 1],
        'Predecessors': ['', 'A', 'A', 'B, C', 'D']
    })

edited_df = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True)

if st.button("Calculate Schedule", type="primary"):
    if '1' not in weekmask:
        st.error("Calculation requires at least one working day selected in the calendar.")
    elif edited_df.empty or edited_df['Activity Name'].isnull().all() or (edited_df['Activity Name'].str.strip() == '').all():
        st.error("Please enter at least one valid activity.")
    else:
        with st.spinner("Calculating Critical Path..."):
            result_df = calculate_cpm(edited_df, start_date, weekmask, holidays)
            
        if result_df is not None:
            st.header("2. Schedule Results")
            
            project_start = result_df['ES'].min()
            project_end = result_df['EF'].max()
            col_sd, col_ed = st.columns(2)
            col_sd.metric("Project Start Date", str(project_start))
            col_ed.metric("Project End Date", str(project_end))
            
            def highlight_critical(row):
                if row.get('Critical Path') == True:
                    return ['background-color: rgba(255, 99, 71, 0.2)'] * len(row)
                return [''] * len(row)
                
            # Convert boolean to a string representation for better UI or keep as boolean checkbox
            st.dataframe(result_df.style.apply(highlight_critical, axis=1), use_container_width=True)
            
            st.header("3. Gantt Chart")
            
            # Prepare data for Plotly Gantt
            gantt_df = result_df.copy()
            # Plotly expects start and end as datetime objects, and for visual accuracy, we add 1 day to end date
            # so the bar covers the entire end date visually.
            gantt_df['Start'] = pd.to_datetime(gantt_df['ES'])
            gantt_df['Finish'] = pd.to_datetime(gantt_df['EF']) + pd.Timedelta(days=1) 
            gantt_df['Task'] = gantt_df['Activity Name']
            
            gantt_df['Type'] = gantt_df['Critical Path'].apply(lambda x: 'Critical (TF=0)' if x else 'Non-Critical')
            
            # Sort by ES ascending so early tasks are at the top
            gantt_df = gantt_df.sort_values(by=['Start', 'Task'], ascending=[True, True])
            
            fig = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Task", color="Type",
                              color_discrete_map={"Critical (TF=0)": "red", "Non-Critical": "blue"},
                              title="Project Schedule Gantt Chart",
                              hover_data={'Duration': True, 'Total Float': True})
            
            # Add vertical rectangles for non-working days and holidays
            if pd.notnull(project_start) and pd.notnull(project_end):
                current_date = pd.to_datetime(project_start)
                end_date = pd.to_datetime(project_end) + pd.Timedelta(days=1)
                
                while current_date <= end_date:
                    # Check if weekend/non-working day or holiday
                    is_working_day = np.is_busday(current_date.date(), weekmask=weekmask, holidays=holidays)
                    if not is_working_day:
                        fig.add_vrect(
                            x0=current_date,
                            x1=current_date + pd.Timedelta(days=1),
                            fillcolor="rgba(150, 150, 150, 0.3)",
                            opacity=1,
                            layer="below",
                            line_width=0,
                        )
                    current_date += pd.Timedelta(days=1)
                              
            fig.update_yaxes(autorange="reversed") # Otherwise tasks are bottom-to-top
            
            st.plotly_chart(fig, use_container_width=True)
