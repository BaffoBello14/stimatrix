-- CTE for ZTL (Limited Traffic Zone) check
-- Checks if parcel centroid is within any ZTL polygon
ZTL_CHECK AS (
    SELECT 
        PC_MAIN.Id as IdParticella,
        CASE 
            WHEN EXISTS (
                SELECT 1 
                FROM ZoneTrafficoLimitato ZTL 
                WHERE ZTL.Poligono.STContains(PC_MAIN.Centroide) = 1
            ) THEN 1 
            ELSE 0 
        END as InZTL
    FROM ParticelleCatastali PC_MAIN
)
