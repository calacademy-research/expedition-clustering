#!/usr/bin/env python3
"""
Simplest possible test - load data from database and run clustering.
"""

import logging
import pymysql
import pandas as pd
import numpy as np
from expedition_clustering import create_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Connecting to database...")
    conn = pymysql.connect(
        host="localhost",
        user="myuser",
        password="mypassword",
        database="exped_cluster_db",
        port=3306,
    )

    logger.info("Loading data with SQL join...")
    query = """
    SELECT
        ce.CollectingEventID as collectingeventid,
        ce.StartDate as startdate,
        ce.EndDate as enddate,
        ce.Remarks as remarks,
        ce.LocalityID as localityid,
        co.CollectionObjectID as collectionobjectid,
        co.Text1 as text1,
        l.MinElevation as minelevation,
        l.MaxElevation as maxelevation,
        l.ElevationAccuracy as elevationaccuracy,
        l.Latitude1 as latitude1,
        l.Longitude1 as longitude1,
        l.LocalityName as localityname,
        l.NamedPlace as namedplace,
        l.GeographyID as geographyid,
        g.CentroidLat as centroidlat,
        g.CentroidLon as centroidlon,
        g.CommonName as commonname,
        g.FullName as fullname,
        g.Name as name
    FROM collectingevent ce
    INNER JOIN collectionobject co ON ce.CollectingEventID = co.CollectingEventID
    LEFT JOIN locality l ON ce.LocalityID = l.LocalityID
    LEFT JOIN geography g ON l.GeographyID = g.GeographyID
    WHERE ce.StartDate IS NOT NULL
      AND (l.Latitude1 IS NOT NULL OR g.CentroidLat IS NOT NULL)
      AND (l.Longitude1 IS NOT NULL OR g.CentroidLon IS NOT NULL)
    LIMIT 10000
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {list(df.columns)}")

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Convert dates
    df['startdate'] = pd.to_datetime(df['startdate'], errors='coerce')
    df['enddate'] = pd.to_datetime(df['enddate'], errors='coerce')

    # Fill in missing coordinates with centroids
    df['latitude1'] = df['latitude1'].fillna(df['centroidlat'])
    df['longitude1'] = df['longitude1'].fillna(df['centroidlon'])

    # Drop rows without coordinates or dates
    df = df[df['latitude1'].notna() & df['longitude1'].notna() & df['startdate'].notna()]

    logger.info(f"After cleaning: {len(df)} rows")
    logger.info(f"Sample data:\n{df[['collectingeventid', 'latitude1', 'longitude1', 'startdate']].head()}")

    # Run clustering
    logger.info("Creating and running clustering pipeline...")
    pipeline = create_pipeline(e_dist=10, e_days=7)
    clustered = pipeline.fit_transform(df)

    logger.info(f"Clustering complete!")
    logger.info(f"Number of unique clusters: {clustered['spatiotemporal_cluster_id'].nunique()}")
    logger.info(f"Average cluster size: {len(clustered) / clustered['spatiotemporal_cluster_id'].nunique():.2f}")
    logger.info(f"\nSample results:\n{clustered[['collectingeventid', 'latitude1', 'longitude1', 'startdate', 'spatial_cluster_id', 'temporal_cluster_id', 'spatiotemporal_cluster_id']].head(20)}")

    logger.info("✓ Test completed successfully!")

if __name__ == "__main__":
    main()
