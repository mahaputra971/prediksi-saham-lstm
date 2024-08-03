/*!999999\- enable the sandbox mode */ 
-- MariaDB dump 10.19-11.4.2-MariaDB, for Linux (x86_64)
--
-- Host: localhost    Database: technical_stock_ta_db
-- ------------------------------------------------------
-- Server version	11.4.2-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*M!100616 SET @OLD_NOTE_VERBOSITY=@@NOTE_VERBOSITY, NOTE_VERBOSITY=0 */;

--
-- Table structure for table `models`
--

DROP TABLE IF EXISTS `models`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `models` (
  `id_model` int(11) NOT NULL AUTO_INCREMENT,
  `id_emiten` int(11) NOT NULL,
  `name` text NOT NULL,
  `created_at` datetime DEFAULT current_timestamp(),
  `model_blob` longblob NOT NULL,
  `algorithm` text NOT NULL,
  `hyperparameters` text DEFAULT NULL,
  `metrics` text DEFAULT NULL,
  PRIMARY KEY (`id_model`),
  KEY `id_emiten` (`id_emiten`),
  CONSTRAINT `models_ibfk_1` FOREIGN KEY (`id_emiten`) REFERENCES `tb_emiten` (`id_emiten`)
) ENGINE=InnoDB AUTO_INCREMENT=38 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tb_accuracy_ichimoku_cloud`
--

DROP TABLE IF EXISTS `tb_accuracy_ichimoku_cloud`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tb_accuracy_ichimoku_cloud` (
  `id_accuracy_ichimoku_cloud` int(11) NOT NULL AUTO_INCREMENT,
  `id_emiten` int(11) NOT NULL,
  `percent_1_hari_sen` float NOT NULL,
  `percent_1_minggu_sen` float NOT NULL,
  `percent_1_bulan_sen` float NOT NULL,
  `percent_1_hari_span` float NOT NULL,
  `percent_1_minggu_span` float NOT NULL,
  `percent_1_bulan_span` float NOT NULL,
  `date` date NOT NULL,
  PRIMARY KEY (`id_accuracy_ichimoku_cloud`),
  KEY `id_emiten` (`id_emiten`),
  CONSTRAINT `tb_accuracy_ichimoku_cloud_ibfk_1` FOREIGN KEY (`id_emiten`) REFERENCES `tb_emiten` (`id_emiten`)
) ENGINE=InnoDB AUTO_INCREMENT=20 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tb_data_ichimoku_cloud`
--

DROP TABLE IF EXISTS `tb_data_ichimoku_cloud`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tb_data_ichimoku_cloud` (
  `id_ichimoku_cloud` int(11) NOT NULL AUTO_INCREMENT,
  `id_emiten` int(11) NOT NULL,
  `tenkan_sen` int(11) NOT NULL,
  `kijun_sen` int(11) NOT NULL,
  `senkou_span_a` int(11) NOT NULL,
  `senkou_span_b` int(11) NOT NULL,
  `date` date NOT NULL,
  PRIMARY KEY (`id_ichimoku_cloud`),
  KEY `tb_ichimoku_cloud_ibfk_1` (`id_emiten`)
) ENGINE=InnoDB AUTO_INCREMENT=14879 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tb_detail_emiten`
--

DROP TABLE IF EXISTS `tb_detail_emiten`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tb_detail_emiten` (
  `id_detail_emiten` int(11) NOT NULL AUTO_INCREMENT,
  `id_emiten` int(11) NOT NULL,
  `date` date NOT NULL,
  `open` int(11) NOT NULL,
  `high` int(11) NOT NULL,
  `low` int(11) NOT NULL,
  `close` int(11) NOT NULL,
  `close_adj` int(11) NOT NULL,
  `volume` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id_detail_emiten`),
  KEY `tb_detail_emiten_ibfk_1` (`id_emiten`)
) ENGINE=InnoDB AUTO_INCREMENT=69622 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tb_emiten`
--

DROP TABLE IF EXISTS `tb_emiten`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tb_emiten` (
  `id_emiten` int(11) NOT NULL AUTO_INCREMENT,
  `kode_emiten` varchar(10) NOT NULL,
  `nama_emiten` varchar(255) NOT NULL,
  `status` tinyint(1) NOT NULL DEFAULT 0,
  PRIMARY KEY (`id_emiten`)
) ENGINE=InnoDB AUTO_INCREMENT=928 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tb_ichimoku_status`
--

DROP TABLE IF EXISTS `tb_ichimoku_status`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tb_ichimoku_status` (
  `id_ichimoku_status` int(11) NOT NULL AUTO_INCREMENT,
  `id_emiten` int(11) NOT NULL,
  `sen_status` varchar(255) NOT NULL,
  `span_status` varchar(255) NOT NULL,
  `date` date NOT NULL,
  PRIMARY KEY (`id_ichimoku_status`),
  KEY `id_emiten` (`id_emiten`)
) ENGINE=InnoDB AUTO_INCREMENT=26 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tb_lstm`
--

DROP TABLE IF EXISTS `tb_lstm`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tb_lstm` (
  `id_lstm` int(11) NOT NULL AUTO_INCREMENT,
  `id_emiten` int(11) NOT NULL,
  `RMSE` float NOT NULL,
  `MAPE` float NOT NULL,
  `MAE` float NOT NULL,
  `MSE` float NOT NULL,
  `date` date NOT NULL,
  PRIMARY KEY (`id_lstm`),
  KEY `tb_lstm_ibfk_1` (`id_emiten`)
) ENGINE=InnoDB AUTO_INCREMENT=27 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tb_prediction_lstm`
--

DROP TABLE IF EXISTS `tb_prediction_lstm`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tb_prediction_lstm` (
  `id_prediction_lstm` int(11) NOT NULL AUTO_INCREMENT,
  `id_emiten` int(11) NOT NULL,
  `max_price` float NOT NULL,
  `min_price` float NOT NULL,
  `max_price_date` date NOT NULL,
  `min_price_date` date NOT NULL,
  `date` date NOT NULL,
  PRIMARY KEY (`id_prediction_lstm`),
  KEY `id_emiten` (`id_emiten`),
  CONSTRAINT `tb_prediction_lstm_ibfk_1` FOREIGN KEY (`id_emiten`) REFERENCES `tb_emiten` (`id_emiten`)
) ENGINE=InnoDB AUTO_INCREMENT=26 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tb_summary`
--

DROP TABLE IF EXISTS `tb_summary`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tb_summary` (
  `id_lstm_summary` int(11) NOT NULL AUTO_INCREMENT,
  `id_emiten` int(11) NOT NULL,
  `pic_closing_price` blob NOT NULL,
  `pic_sales_volume` blob NOT NULL,
  `pic_price_history` blob NOT NULL,
  `pic_comparation` blob NOT NULL,
  `pic_prediction` blob NOT NULL,
  `pic_ichimoku_cloud` blob NOT NULL,
  `render_date` date NOT NULL,
  PRIMARY KEY (`id_lstm_summary`),
  KEY `tb_lstm_summary_ibfk_1` (`id_emiten`)
) ENGINE=InnoDB AUTO_INCREMENT=20 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping routines for database 'technical_stock_ta_db'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*M!100616 SET NOTE_VERBOSITY=@OLD_NOTE_VERBOSITY */;

-- Dump completed on 2024-08-02  1:49:52
