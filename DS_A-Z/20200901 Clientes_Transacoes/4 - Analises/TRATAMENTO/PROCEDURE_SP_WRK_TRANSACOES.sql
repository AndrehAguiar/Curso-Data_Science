-- ================================================
-- Template generated from Template Explorer using:
-- Create Procedure (New Menu).SQL
--
-- Use the Specify Values for Template Parameters 
-- command (Ctrl-Shift-M) to fill in the parameter 
-- values below.
--
-- This block of comments will not be included in
-- the definition of the procedure.
-- ================================================
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:	Andre Aguiar
-- Create date: 20200901
-- Description:	PROCEDURE QUE CRIA A TAB WRK_TRANSACOES
-- =============================================
CREATE PROCEDURE SP_WRK_TRANSACOES

AS
BEGIN

-- =============================================
-- DROP TABLE
-- =============================================
IF OBJECT_ID('WRK_TRANSACOES') IS NOT NULL
DROP TABLE [dbo].[WRK_TRANSACOES]

-- =============================================
-- CREATE TABLE
-- =============================================
CREATE TABLE WRK_TRANSACOES
(
		[RowNumber]			INT IDENTITY
		,[Order ID]			VARCHAR(100)
		,[Order Date]		DATE
		,[Customer ID]		VARCHAR(100)
		,[Region]			CHAR(10)
		,[Rep]				VARCHAR(100)
		,[Item]				VARCHAR(100)
		,[Units]			INT
		,[Unit Price]		FLOAT

)

-- =============================================
-- TRUNCATE TABLE
-- =============================================
TRUNCATE TABLE [dbo].[WRK_TRANSACOES]

-- =============================================
-- INSERT DATA
-- =============================================
INSERT INTO [dbo].[WRK_TRANSACOES]
(
		[Order ID]
		,[Order Date]
		,[Customer ID]
		,[Region]
		,[Rep]
		,[Item]
		,[Units]
		,[Unit Price]
)
SELECT 
		[Order ID]
		,[Order Date]
		,[Customer ID]
		,[Region]
		,[Rep]
		,[Item]
		,[Units]
		,[Unit Price]

  FROM [DS_TRAININIG].[dbo].[RAW_Transacoes_20200901]

-- ===============================================
-- (43 row(s) affected)
-- ===============================================
END
GO
