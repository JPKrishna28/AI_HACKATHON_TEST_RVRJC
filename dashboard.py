import streamlit as st
import psycopg2
import pandas as pd
import base64

# Database config
DB_CONFIG = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": "1234",
    "port": 5432
}

def get_data():
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT t.id, a.filename, t.transcript, t.threat_type, t.audio_file
        FROM threat_analysis_results t
        JOIN audio_input_table a ON a.id = t.audio_id
        ORDER BY t.id DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def audio_bytes_to_base64(audio_bytes):
    return base64.b64encode(audio_bytes).decode('utf-8')

def main():
    st.title("Threat Detection Dashboard")

    # Load data
    df = get_data()

    # Filter by threat type
    threat_types = df['threat_type'].unique().tolist()
    selected_threats = st.multiselect("Filter by Threat Type:", threat_types, default=threat_types)

    filtered_df = df[df['threat_type'].isin(selected_threats)]

    # Search bar
    search_query = st.text_input("Search in Transcripts:")
    if search_query:
        filtered_df = filtered_df[filtered_df['transcript'].str.contains(search_query, case=False, na=False)]

    # Display results
    st.markdown(f"### Showing {len(filtered_df)} records")

    for _, row in filtered_df.iterrows():
        with st.expander(f"ID {row['id']} - {row['filename']} ({row['threat_type']})"):
            st.write(row['transcript'])

            # Play audio
            if row['audio_file']:
                b64 = audio_bytes_to_base64(row['audio_file'])
                audio_html = f'''
                    <audio controls>
                        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                '''
                st.markdown(audio_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
