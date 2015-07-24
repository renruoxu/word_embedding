import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vectors, Matrices}
import scala.tools.nsc.io.File 

val matrixFile = "data/tmp/pmi.txt"

def loadMatrix(matrixFile: String): RowMatrix = {
	val rawRdd = sc.textFile(matrixFile)
	// first line as shape of matrix
	val header = rawRdd.first.split(" ")
	val nrows = header(1).toInt
	val ncols = header(2).toInt
	// load r, c, v each line
	val matrixData = rawRdd.filter(!_.contains("#")).map {line =>
		val fields = line.split(",")
		val row = fields(0).toInt
		val col = fields(1).toInt 
		val value = fields(2).toDouble
		( row, (col, value) )
	}
	val rowvecs = matrixData.groupByKey().sortBy(_._1).map {case (r, cols) => 
		Vectors.sparse(ncols, cols.toSeq)
	}
	rowvecs.cache()
	new RowMatrix(rowvecs)
}

val pmiMatrix = loadMatrix(matrixFile)

val dim = 200

val svd = pmiMatrix.computeSVD(dim, computeU= true)

def saveMatrix(matrix: RowMatrix, saveFile: String): Unit = {
	val rowStrs = matrix.rows.map { row => 
		row.toArray.mkString(",")
	}
	val matStr = rowStrs.collect.mkString("\n")
	File(saveFile).writeAll(matStr)
}

saveMatrix(svd.U, "data/tmp/U.txt")


def saveVector(vector: org.apache.spark.mllib.linalg.Vector, saveFile: String): Unit = {
	File(saveFile).writeAll(vector.toArray.mkString(","))
}

saveVector(svd.s, "data/tmp/s.txt")