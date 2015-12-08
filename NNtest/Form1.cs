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
            textBox1.Text = eta.ToString();
            textBox2.Text = alpha.ToString();
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
            }
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
        //2次元データプロット
        void plot(Chart chart,double[][] data,bool update_flag)
        {
            if(data[0].Length==2)
            {
                Series series = new Series();
                for (int n = 0; n < data.Length; n++)
                    series.Points.AddXY(data[n][0], data[n][1]);
                series.Color = Color.Blue;
                series.MarkerColor = Color.Blue;
                series.MarkerSize = 10;
                series.MarkerStyle = System.Windows.Forms.DataVisualization.Charting.MarkerStyle.Circle;
                series.ChartType = SeriesChartType.Point;
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
            init();
            update(100);
        }

        private void textBox1_TextChanged_1(object sender, EventArgs e)
        {
            if (textBox1.Text != null)
                eta = double.Parse(textBox1.Text);

        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {
            if (textBox2.Text != null)
                alpha = double.Parse(textBox2.Text);
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox1.Checked)
                bias_flag = true;
            else
                bias_flag = false;
        }
    }


}
