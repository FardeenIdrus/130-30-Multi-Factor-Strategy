-- create schema
CREATE SCHEMA IF NOT EXISTS systematic_equity AUTHORIZATION postgres;

-- Equity Static
CREATE TABLE IF NOT EXISTS fift.systematic_equity.company_static (
	"symbol" CHAR(12) PRIMARY KEY,
	"security" TEXT,
	"gics_sector"	TEXT,
	"gics_industry"	TEXT,
	"country"	TEXT,
	"region"	TEXT	
);