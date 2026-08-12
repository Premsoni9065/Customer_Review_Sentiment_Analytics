-- ============================================================
-- review_analysis.sql
-- Customer Review & Sentiment Analytics
-- Beginner-level SQL analysis queries
--
-- Assumes a table called `customer_reviews` with columns:
-- Review_ID, Product_ID, Product_Name, Category, Customer_ID,
-- Review_Date, Rating, Review_Title, Review_Text,
-- Verified_Purchase, Helpful_Votes, Sentiment
--
-- (Sentiment is added by the Python/NLTK VADER step in app.py and can
--  be loaded into this table before running the queries below.)
-- Developed & Deployed by Prem Kumar
-- ============================================================


-- 1. Total number of reviews
SELECT COUNT(*) AS total_reviews
FROM customer_reviews;


-- 2. Average rating across all reviews
SELECT ROUND(AVG(Rating), 2) AS average_rating
FROM customer_reviews;


-- 3. Number of reviews for each star rating (1 to 5)
SELECT Rating, COUNT(*) AS review_count
FROM customer_reviews
GROUP BY Rating
ORDER BY Rating;


-- 4. Total positive reviews
SELECT COUNT(*) AS positive_reviews
FROM customer_reviews
WHERE Sentiment = 'Positive';


-- 5. Total negative reviews
SELECT COUNT(*) AS negative_reviews
FROM customer_reviews
WHERE Sentiment = 'Negative';


-- 6. Total neutral reviews
SELECT COUNT(*) AS neutral_reviews
FROM customer_reviews
WHERE Sentiment = 'Neutral';


-- 7. Average rating by category
SELECT Category, ROUND(AVG(Rating), 2) AS avg_rating
FROM customer_reviews
GROUP BY Category
ORDER BY avg_rating DESC;


-- 8. Review count by category
SELECT Category, COUNT(*) AS review_count
FROM customer_reviews
GROUP BY Category
ORDER BY review_count DESC;


-- 9. Top 10 rated products (minimum 5 reviews to avoid outliers)
SELECT Product_Name,
       COUNT(*) AS review_count,
       ROUND(AVG(Rating), 2) AS avg_rating
FROM customer_reviews
GROUP BY Product_Name
HAVING COUNT(*) >= 5
ORDER BY avg_rating DESC
LIMIT 10;


-- 10. Lowest 10 rated products (minimum 5 reviews to avoid outliers)
SELECT Product_Name,
       COUNT(*) AS review_count,
       ROUND(AVG(Rating), 2) AS avg_rating
FROM customer_reviews
GROUP BY Product_Name
HAVING COUNT(*) >= 5
ORDER BY avg_rating ASC
LIMIT 10;


-- 11. Verified vs non-verified purchase review counts and average rating
SELECT Verified_Purchase,
       COUNT(*) AS review_count,
       ROUND(AVG(Rating), 2) AS avg_rating
FROM customer_reviews
GROUP BY Verified_Purchase;


-- 12. Average helpful votes by rating
SELECT Rating, ROUND(AVG(Helpful_Votes), 2) AS avg_helpful_votes
FROM customer_reviews
GROUP BY Rating
ORDER BY Rating;


-- ============================================================
-- BONUS QUERIES (extra beginner-friendly practice)
-- ============================================================

-- 13. Sentiment breakdown (%) per category using CASE WHEN
SELECT
    Category,
    COUNT(*) AS total_reviews,
    ROUND(100.0 * SUM(CASE WHEN Sentiment = 'Positive' THEN 1 ELSE 0 END) / COUNT(*), 2) AS positive_pct,
    ROUND(100.0 * SUM(CASE WHEN Sentiment = 'Neutral' THEN 1 ELSE 0 END) / COUNT(*), 2) AS neutral_pct,
    ROUND(100.0 * SUM(CASE WHEN Sentiment = 'Negative' THEN 1 ELSE 0 END) / COUNT(*), 2) AS negative_pct
FROM customer_reviews
GROUP BY Category
ORDER BY positive_pct DESC;


-- 14. Products with the highest share of negative reviews
--     (useful for the "Customer Complaints" section of the dashboard)
SELECT
    Product_Name,
    COUNT(*) AS total_reviews,
    SUM(CASE WHEN Sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative_reviews,
    ROUND(100.0 * SUM(CASE WHEN Sentiment = 'Negative' THEN 1 ELSE 0 END) / COUNT(*), 2) AS negative_pct
FROM customer_reviews
GROUP BY Product_Name
HAVING COUNT(*) >= 5
ORDER BY negative_pct DESC
LIMIT 10;


-- 15. Monthly review volume trend
SELECT
    STRFTIME('%Y-%m', Review_Date) AS review_month,
    COUNT(*) AS review_count
FROM customer_reviews
GROUP BY review_month
ORDER BY review_month;
