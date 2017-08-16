import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{SparseVector,Matrices,Vectors}
import breeze.linalg._
import breeze.numerics._
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.qr
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrix

object SSVDFinal {

def main(args: Array[String]) {
	val conf = new SparkConf()
				.setAppName("SSVDFinal")
				.setMaster("local[*]")
	    val sc = new SparkContext(conf)
var stopWordsArray = Array("rt","a","an","the","able","about","after","all","almost", "also","am","and","any","are","as","at","be","because","been","but","by", "can","cannot","could","dear","did","do","does","don't","either","else","ever","every","for","from","get","got","had","has","have","he","her","him","his","how","however","if","in","into","is", "it","like","may","me","might","most", "must","my","no","nor","not","of","on","only","or","other","our", "own","rather","said","say","she","should","since","so","some","than","that","the","their","them","then","there","these", "they","this","to","too","was","us","was", "we","were","what","when", "where","which","while","who","whom","why","will","with","would","yet","you", "your")

def stopFunc(splitData: Array[String]) : Array[String] = {
var stopArrayBuffer = new ArrayBuffer[String]
for(i<-0 to splitData.length-1) {
if(!(stopWordsArray.contains(splitData(i)))) 
stopArrayBuffer +=splitData(i)
}
var noStopArray = new Array[String](stopArrayBuffer.size)
stopArrayBuffer.copyToArray(noStopArray)
noStopArray
}
//val twitterStopWordsData  = sc.textFile("/Users/Alya/Desktop/SSVD/TestData/Twit.txt")
//val tweetDataStopWords  = sc.textFile("/Users/Alya/Desktop/SSVD/TestData/Twit.txt")

val tweetDataStopWords  = sc.textFile("/Users/Alya/Desktop/SSVD/TestData/Twitt/part-00001")
val twitterStopWordsData  = sc.textFile("/Users/Alya/Desktop/SSVD/TestData/Twitt/part-00001")

var stopWordstest = tweetDataStopWords.map(line=>line.split("\t")).map(r=>r(1).toLowerCase()).map(data=>data.split(" ")).flatMap(a=>stopFunc(a)).distinct().zipWithIndex

var zippedDataStopWords = stopWordstest.collect

var uniqueWordsStop = stopWordstest.count


def joinFunc(val1: Array[String]) : org.apache.spark.mllib.linalg.SparseVector = {
var throwMap = collection.mutable.Map[Long, org.apache.spark.mllib.linalg.Vector]()
var nonZeroIndices = new Array[Int](val1.length-1) 
var prev = 0
for(i<-0 to val1.length-1) 
{
if(i!=0) 
{
var a = zippedDataStopWords.filter { case(key, value) => key.equals(val1(i).trim) }.map(p=>p._2)
nonZeroIndices(i-1) = a(0).toInt
}
}
scala.util.Sorting.quickSort(nonZeroIndices)
var uniqueNonZero  = new Array[Int](nonZeroIndices.distinct.length)
var nonzeroWordCounts = new Array[Double](nonZeroIndices.distinct.length)
nonZeroIndices.distinct.copyToArray(uniqueNonZero)
var occur = 0
var temp = -1
var k = -1
for(j<-0 to nonZeroIndices.length-1) {
if(temp == nonZeroIndices(j))
{
occur = occur + 1
nonzeroWordCounts(k) = occur
}
else {
k = k+1
occur=1
nonzeroWordCounts(k) = occur
temp = nonZeroIndices(j)
}
}
var sparVect = new org.apache.spark.mllib.linalg.SparseVector(uniqueWordsStop.toInt,uniqueNonZero,nonzeroWordCounts)
sparVect
}

var matrixData = twitterStopWordsData.map(line=>(line.toLowerCase().split("\t")).flatMap(data=>data.split(" "))).map(a=>stopFunc(a)).map(p=>joinFunc(p))

val k = 10
val p = 5
var OrigDenseMat = new breeze.linalg.DenseMatrix(matrixData.count.toInt,uniqueWordsStop.toInt,matrixData.collect.flatMap(a=>a.toArray))
val breeze.linalg.svd.SVD(uOrig,sOrig,vOrig) = breeze.linalg.svd(OrigDenseMat)
var SOrigDensMat = diag(sOrig)
//var SVDOrig = uOrig(0 until matrixData.count.toInt,0 until k) * SOrigDensMat(0 until k,0 until k) * ((vOrig.t)(0 until k,0 until matrixData.count.toInt))

var SVDOrig = uOrig(::,0 until k) * SOrigDensMat(0 until k,0 until k) * ((vOrig.t)(0 until k,::))
var FrobeniousSVDMat = SVDOrig +(OrigDenseMat :*(-1.0))

var FrobNormSVD = norm(FrobeniousSVDMat.toDenseVector)/k.toDouble



//SVD of Original Matrix done till here

/*var ind  = matrixData.map{a=>var b = a.indices
						var RandVect = DenseVector.rand(uniqueWordsStop.toInt)
						for(i<-0 until b.length) { 
						b.map(d=>(b,(a.apply(b(i)) * RandVect(b(i)))))
						 }
						}
*/

var denseRand = DenseMatrix.rand(uniqueWordsStop.toInt, k+p)

//var denseRand = new DenseMatrix(uniqueWordsStop.toInt, k+p,Array(0.5,0.75,0.25,0.35,0.45,0.3,0.20,0.9,0.2,0.1))
var broadCastMat = sc.broadcast(denseRand)

var ind  = matrixData.map{a=>
						var b = a.indices
						var c = new Array[Double](k+p)
						var prev=0.0; 
						for(j<-0 until k+p) { 
						for(i<-0 until b.length) { 
						prev = (a.apply(b(i)) * broadCastMat.value(b(i),j))+ prev 
						}
						c(j) = prev 
						prev = 0 } 
						Vectors.dense(c) }
//RIGHT NOW
//var densMatTransp = new DenseMatrix(ind.count.toInt, (k+p),ind.flatMap(s=>s.toArray).collect)
var densMatTransp = new DenseMatrix((k+p),ind.count.toInt,ind.flatMap(s=>s.toArray).collect)
var qmatrix = qr.justQ(densMatTransp.t)
//for(i<-0 to 1) {
var YDenseMat = densMatTransp.t * densMatTransp * qmatrix
var newQMat = qr.justQ(YDenseMat)
//}
var BMatrix = newQMat.t * densMatTransp.t
val breeze.linalg.svd.SVD(uSSVD,sSSVD,vSSVD) = breeze.linalg.svd(BMatrix.t* BMatrix)
var sigmaDensMat = diag(DenseVector(sSSVD.toArray.map(a=>1/Math.sqrt(a))))

var QmatChopped = newQMat(::,0 until k+p)
var UCHopped = uSSVD(::,0 until k)
var UFinal = QmatChopped * UCHopped
//V Job
var VTransp = sigmaDensMat * vSSVD * BMatrix.t
var VFinal = VTransp.t(::,0 until k)

var SVDMulStoch = UFinal*sigmaDensMat(0 until k,0 until k)*VFinal.t  
println("DOne till SVDMULSTOC And rows= "+SVDMulStoch.rows+" And cols= "+SVDMulStoch.cols)

var FrobeniousSSVDMat = OrigDenseMat(::,0 until SVDMulStoch.cols) +(SVDMulStoch :*(-1.0))

var FrobNormSSVD = norm(FrobeniousSSVDMat.toDenseVector)/k.toDouble

println("NUMBER OF ROWS= "+matrixData.count)
println("NUMBER OF COLS= "+uniqueWordsStop)
//println("SVD on original Matrix = "+SVDOrig)
//println("SSVD on preconditioned= "+SVDMulStoch)

println("FROBENOIS NORM= "+FrobNorm)






/*var densMat = new DenseMatrix(ind.count.toInt, (k+p),ind.flatMap(s=>s.toArray).collect)
var qmatrix = qr.justQ(densMat)
var YDenseMat = densMat * densMat.t * qmatrix
var newQMat = qr.justQ(YDenseMat)
var BMatrix = newQMat.t * densMat
val breeze.linalg.svd.SVD(uSSVD,sSSVD,vSSVD) = breeze.linalg.svd(BMatrix* BMatrix.t)
var sigmaDensMat = diag(DenseVector(sSSVD.toArray.map(a=>1/Math.sqrt(a))))
var VTransp = sigmaDensMat * vSSVD * BMatrix
var QmatChopped = newQMat(::,0 until k+p)
var UCHopped = uSSVD(::,0 until k)
var UFinal = QmatChopped * UCHopped
var VFinal = VTransp.t(::,0 until k)

var SVDMulStoch = UFinal*sigmaDensMat(0 until k,0 until k)*VFinal.t 
*/






}
}

