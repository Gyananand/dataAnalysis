SELECT *
FROM layoffs_staging2;


SELECT MAX(total_laid_off),
       MAX(percentage_laid_off)
FROM layoffs_staging2;


SELECT *
FROM layoffs_staging2
WHERE percentage_laid_off = 1
ORDER BY funds_raised_millions DESC;


SELECT company,
       SUM(total_laid_off)
FROM layoffs_staging2
GROUP BY company
ORDER BY 2 DESC;


SELECT industry,
       SUM(total_laid_off)
FROM layoffs_staging2
GROUP BY industry
ORDER BY 2 DESC;


SELECT country,
       SUM(total_laid_off)
FROM layoffs_staging2
GROUP BY country
ORDER BY 2 DESC;


SELECT YEAR(`date`),
       SUM(total_laid_off)
FROM layoffs_staging2
GROUP BY YEAR(`date`)
ORDER BY 1 DESC;


SELECT stage,
       SUM(total_laid_off)
FROM layoffs_staging2
GROUP BY stage
ORDER BY 2 DESC;


SELECT substring(`date`, 1, 7) AS `MONTH`,
       SUM(total_laid_off)
FROM layoffs_staging2
WHERE substring(`date`, 1, 7) IS NOT NULL
GROUP BY `MONTH`
ORDER BY 1 ASC;

WITH ROLLING_TOTAL AS
  (SELECT substring(`date`, 1, 7) AS `MONTH`,
          SUM(total_laid_off) total_off
   FROM layoffs_staging2
   WHERE substring(`date`, 1, 7) IS NOT NULL
   GROUP BY `MONTH`
   ORDER BY 1 ASC)
SELECT `MONTH`,
       total_off,
       SUM(total_off) OVER(
                           ORDER BY `MONTH`) AS rolling_total
FROM ROLLING_TOTAL;


SELECT company,
       YEAR(`date`),
       SUM(total_laid_off)
FROM layoffs_staging2
GROUP BY company,
         YEAR(`date`)
ORDER BY 3 DESC;

WITH Company_Year (COMPANY, YEARS, TOTAL_LAID_OFF) AS
  (SELECT company,
          YEAR(`date`),
          SUM(total_laid_off)
   FROM layoffs_staging2
   GROUP BY company,
            YEAR(`date`)),
     COMPANY_YEAR_RANK AS
  (SELECT *,
          DENSE_RANK() OVER(PARTITION BY years
                            ORDER BY total_laid_off DESC) AS RANKING
   FROM Company_Year
   WHERE YEARS IS NOT NULL)
SELECT *
FROM COMPANY_YEAR_RANK
WHERE RANKING <= 5;