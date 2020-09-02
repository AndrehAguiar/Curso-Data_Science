/***** CREATE TABLE [DRV_CLIENTESTRANSACOES], INSERT DATA FROM dbo.WRK_CLIENTES INNER JOIN dbo.WRK_TRANSACOES *****/

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
-- Description:	PROCEDURE QUE CRIA A TAB WRK_CLIENTES
-- =============================================
CREATE PROCEDURE [dbo].[SP_DRV_CLIENTESTRANSACOES]

AS
BEGIN

-- =============================================
-- DROP TABLE
-- =============================================
IF OBJECT_ID('DRV_CLIENTESTRANSACOES') IS NOT NULL
DROP TABLE [dbo].[DRV_CLIENTESTRANSACOES]

-- =============================================
-- CREATE TABLE
-- =============================================
CREATE TABLE [dbo].[DRV_CLIENTESTRANSACOES]
(
		[RowNumber]			int identity
		,[Customer ID]		NUMERIC
		,[City]				VARCHAR(100)
		,[ZipCode]			VARCHAR(10)
		,[Gender]			CHAR(1)
		,[Age]				FLOAT
		,[Order ID]			VARCHAR(100)
		,[Order Date]		DATE
		,[Region]			CHAR(10)
		,[Rep]				VARCHAR(100)
		,[Item]				VARCHAR(100)
		,[Units]			NUMERIC
		,[Unit Price]		FLOAT
)

-- =============================================
-- TRUCATE TABLE
-- =============================================
TRUNCATE TABLE [dbo].[DRV_CLIENTESTRANSACOES]

-- =============================================
-- INSERT DATA
-- =============================================
INSERT INTO [dbo].[DRV_CLIENTESTRANSACOES]
(
		[Customer ID]
		,[City]
		,[ZipCode]
		,[Gender]
		,[Age]
		,[Order ID]
		,[Order Date]
		,[Region]
		,[Rep]
		,[Item]
		,[Units]
		,[Unit Price]
)
SELECT
		dbo.WRK_CLIENTES.[Customer ID],
		dbo.WRK_CLIENTES.City,
		dbo.WRK_CLIENTES.ZipCode,
		dbo.WRK_CLIENTES.Gender,
		dbo.WRK_CLIENTES.Age,
		dbo.WRK_TRANSACOES.[Order ID], 
		dbo.WRK_TRANSACOES.[Order Date],
		dbo.WRK_TRANSACOES.Region,
		dbo.WRK_TRANSACOES.Rep,
		dbo.WRK_TRANSACOES.Item,
		dbo.WRK_TRANSACOES.Units,
		dbo.WRK_TRANSACOES.[Unit Price]
FROM
dbo.WRK_CLIENTES INNER JOIN	dbo.WRK_TRANSACOES
ON dbo.WRK_CLIENTES.[Customer ID] = dbo.WRK_TRANSACOES.[Customer ID]
-- =======================================================
-- (43 row(s) affected)
-- =======================================================

END
GO
