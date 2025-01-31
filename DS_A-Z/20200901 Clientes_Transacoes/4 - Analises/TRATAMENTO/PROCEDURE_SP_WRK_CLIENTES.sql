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

USE DS_TRAININIG
GO

-- =============================================
-- Author:		Andre Aguiar
-- Create date: 20200901
-- Description:	PROCEDURE QUE CRIA A TAB WRK_CLIENTES
-- =============================================
ALTER PROCEDURE SP_WRK_CLIENTES

AS
BEGIN
-- ===============================================
-- DROP TABLE
-- ===============================================
IF OBJECT_ID('WRK_CLIENTES') IS NOT NULL
DROP TABLE [dbo].[WRK_CLIENTES]

-- ===============================================
-- CREATE TABLE
-- ===============================================
CREATE TABLE WRK_CLIENTES
(
	 [RowNumber]		INT IDENTITY
	,[Customer ID]		VARCHAR(100)
	,[City]				VARCHAR(100)
	,[ZipCode]			VARCHAR(10)
	,[Gender]			CHAR(1)
	,[Age]				FLOAT
)

-- ===============================================
-- TRUNCATE TABLE
-- ===============================================
TRUNCATE TABLE [dbo].[WRK_CLIENTES]

-- ===============================================
-- INSERT DATA
-- ===============================================

INSERT INTO [dbo].[WRK_CLIENTES]
(			[Customer ID]
           ,[City]
           ,[ZipCode]
           ,[Gender]
           ,[Age])
SELECT
		 [Customer ID]
		,[City]
		,[ZipCode]
		,[Gender]
		,[Age]
  FROM [dbo].[RAW_Clientes_20200901]

-- ===============================================
-- (43 row(s) affected)
-- ===============================================
END
GO
