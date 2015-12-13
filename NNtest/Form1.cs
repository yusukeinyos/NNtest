using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using InoueLab;
using System.Windows.Forms.DataVisualization.Charting;

namespace NNtest
{
    public partial class Form1 : Form
    {
        double[][] data_set; //dataset(X,class label t)
        double[][] X; //input data  N×I matrix
        double[] w; //weight parameter  N×I matrix
        double[] t;  //target  1×I vector

        double eta; //learning rate
        double alpha; //weight decay

        int N; //number of datasets
        int I; //dimention of data
        int itteration;

        bool bias_flag;

        public Form1()
        {
            InitializeComponent();
            init();
        }
        //-----------------------------------------------------------------------------
        void init()
        {
            readData("data_XandT.csv");
            t = New.Array(data_set.Length, n => data_set[n][data_set[0].Length - 1]);
            N = t.Length;
            eta = 0.01;
            alpha = 0.01;
            bias_flag = true;
            itteration = 100;
            textBox1.Text = eta.ToString();
            textBox2.Text = alpha.ToString();
            textBox3.Text = itteration.ToString();
            checkBox1.Checked = true;
        }
        //-----------------------------------------------------------------------------
        //NN classification(Binary label) with single neuron
        void update(int itteration)
        {
            double[] a;
            double[] y;
            double[] e;
            double[] g;
            double error;
            if (bias_flag)
            {
                X = New.Array(data_set.Length, n => New.Array(data_set[0].Length, i => data_set[n][i]));
                foreach (double[] d in X)
                    d[data_set[0].Length - 1] = 1.0;
            }
            else
                X = New.Array(data_set.Length, n => New.Array(data_set[0].Length - 1, i => data_set[n][i]));
            plot(chart1);
            I = X[0].Length;
            w = new double[I];
            for (int i = 0; i < I; i++)
                chart2.Series[i].Points.Clear();
            chart3.Series[0].Points.Clear();
            listBox1.Items.Clear();
            for (int ii = 0; ii < itteration; ii++)
            {
                //compute all activations
                a = New.Array(N, i => Mt.Inner(X[i], w));
                //compute outputs
                y = New.Array(N, i => sigmoid(a[i]));
                //compute errors
                e = Mt.Sub(t, y);
                error = Mt.Inner(e, e);
                listBox1.Items.Add("itteration : " + (ii + 1) + ", error = " + error);
                //compute the gradient vector
                double[][] xt = New.Array(I, i => New.Array(N, n => X[n][i]));
                g = New.Array(I, i => Mt.Inner(xt[i], e)).Neg();
                //make step, using learning rate eta and weight decay alpha
                w = Mt.Sub(w, Mt.Mul(eta, Mt.Add(g, Mt.Mul(alpha, w))));

                plot(chart2, chart2.Series[0], w[0], SeriesChartType.Spline, Color.YellowGreen); //dictionaryにより色とintを対応
                plot(chart2, chart2.Series[1], w[1], SeriesChartType.Spline, Color.Red);
                if (I == 3)
                {
                    plot(chart2, chart2.Series[2], w[2], SeriesChartType.Spline, Color.Blue);
                    chart2.Series[2].Enabled = true;
                }
                else
                    chart2.Series[2].Enabled = false;
                plot(chart3, chart3.Series[0], Mt.Inner(w, w) / 2.0, SeriesChartType.Spline, Color.YellowGreen);
            }
            boundary();
        }
        //-----------------------------------------------------------------------------
        //決定境界
        void boundary()
        {
            double[][] boundary;
            int bin = 100;
            boundary = New.Array(bin, n => New.Array(2, i => 0.0));
            double[] x = New.Array(N, n => X[n][0]);
            double x_min = x.Min();
            double x_max = x.Max();
            double interval = (double)(x_max - x_min) / bin;

            for (int i = 0; i < bin; i++)
            {
                boundary[i][0] = x_min + interval * i;
                if (I == 3)
                    boundary[i][1] = -(w[0] * boundary[i][0] + w[2]) / w[1];
                else
                    boundary[i][1] = -(w[0] * boundary[i][0]) / w[1];
            }
            plot(chart1, boundary, true, SeriesChartType.Spline, Color.White);

        }
        //-----------------------------------------------------------------------------
        double sigmoid(double d)
        {
            return 1.0 / (1.0 + Math.Exp(-d));
        }
        //-----------------------------------------------------------------------------
        void readData(string filename)
        {
            using (StreamReader stream = new StreamReader(filename))
            {
                string all = stream.ReadToEnd();
                var lines = all.Split(new char[] { '\n' }).Where(l => l != "").ToArray();
                data_set = new double[lines.Length][];
                for (int i = 0; i < lines.Length; i++)
                {
                    var line = lines[i].Split(new char[] { ',' }).Where(a => a != "").ToArray();
                    data_set[i] = new double[line.Length];
                    for (int j = 0; j < line.Length; j++)
                    {
                        data_set[i][j] = double.Parse(line[j]);
                    }
                }
            }

        }
        //-----------------------------------------------------------------------------
        void plot(Chart chart)
        {
            Series series_data_label1 = new Series("class 1");
            Series series_data_label2 = new Series("class 2");

            for (int n = 0; n < N; n++)
            {
                if (data_set[n][2] == 1.0)
                    series_data_label1.Points.AddXY(X[n][0], X[n][1]);
                else
                    series_data_label2.Points.AddXY(X[n][0], X[n][1]);
            }
            {
                series_data_label1.Color = Color.Blue;
                series_data_label1.MarkerColor = Color.Blue;
                series_data_label1.MarkerSize = 10;
                series_data_label1.MarkerStyle = System.Windows.Forms.DataVisualization.Charting.MarkerStyle.Circle;
                series_data_label1.ChartType = SeriesChartType.Point;
                series_data_label1.Color = Color.Red;
                series_data_label1.MarkerColor = Color.Red;
                series_data_label1.MarkerSize = 10;
                series_data_label1.MarkerStyle = System.Windows.Forms.DataVisualization.Charting.MarkerStyle.Cross;
                series_data_label2.ChartType = SeriesChartType.Point;
            }
            chart.Series.Clear();
            chart.Series.Add(series_data_label1);
            chart.Series.Add(series_data_label2);
            chart.ChartAreas[0].BackColor = Color.Black;
            chart.ChartAreas[0].BackImageTransparentColor = Color.Black;

        }
        //-----------------------------------------------------------------------------
        //2次元データプロット(データ追加式)
        void plot(Chart chart, Series series, double data, SeriesChartType st, Color color)
        {
            series.Points.AddY(data);
            series.Color = color;
            series.MarkerColor = color;
            series.MarkerSize = 2;
            series.MarkerStyle = System.Windows.Forms.DataVisualization.Charting.MarkerStyle.Circle;
            series.ChartType = st;
            //chart.Series.Clear();
            //chart.Series.Add(series);
            chart.ChartAreas[0].BackColor = Color.Black;
            chart.ChartAreas[0].BackImageTransparentColor = Color.Black;
        }
        //-----------------------------------------------------------------------------
        //2次元データプロット
        void plot(Chart chart, double[][] data, bool update_flag, SeriesChartType st, Color color)
        {
            if (data[0].Length == 2)
            {
                Series series = new Series();
                for (int n = 0; n < data.Length; n++)
                    series.Points.AddXY(data[n][0], data[n][1]);
                series.Color = color;
                series.MarkerColor = color;
                series.MarkerSize = 2;
                series.MarkerStyle = System.Windows.Forms.DataVisualization.Charting.MarkerStyle.Circle;
                series.ChartType = st;
                if (!update_flag)
                    chart.Series.Clear();
                chart.Series.Add(series);
                chart.ChartAreas[0].BackColor = Color.Black;
                chart.ChartAreas[0].BackImageTransparentColor = Color.Black;
            }
        }
        //-----------------------------------------------------------------------------
        private void button1_Click_1(object sender, EventArgs e)
        {
            //init();
            update(itteration);
        }

        private void textBox1_TextChanged_1(object sender, EventArgs e)
        {
            if (textBox1.Text != "")
                eta = double.Parse(textBox1.Text);

        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {
            if (textBox2.Text != "")
                alpha = double.Parse(textBox2.Text);
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox1.Checked)
                bias_flag = true;
            else
                bias_flag = false;
        }

        private void textBox3_TextChanged(object sender, EventArgs e)
        {
            if (textBox3.Text != "")
                itteration = int.Parse(textBox3.Text);
        }
    }


}
