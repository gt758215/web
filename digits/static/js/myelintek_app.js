'use strict';

try {
(function() {
    var app = angular.module('mlt_app', [])

    app.controller('models_controller', function($scope, $controller) {
        $controller('job_controller', {$scope: $scope});
        $scope.title = 'Models';
        $scope.model_fields = [
            {name: 'Name', show: true, min_width: 100},
            {name: 'id', show: false, min_width: 200},
            {name: 'Status', show: true, min_width: 50},
            {name: 'Accuracy', show: true, min_width: 50},
            {name: 'Progress', show: true, min_width: 50},
            {name: 'Elapsed', show: true, min_width: 50},
            {name: 'Submitted', show: true, min_width: 50}
        ];

        $scope.print_time_diff_terse = function(diff) {
            return print_time_diff_terse(diff);
        };
    });

    app.controller('evaluations_controller', function($scope, $controller){
        $controller('job_controller', {$scope: $scope});
        $scope.title = 'Evaluations';
        $scope.fields = [{name: 'Name', show: true, min_width: 100},
            {name: 'Model', show: true, min_width: 100},
            {name: 'Dataset', show: true, min_width: 100},
            {name: 'Status', show: true, min_width: 100},
            {name: 'Progress', show: true, min_width: 100},
            {name: 'Elapsed', show: true, min_width: 100},
            {name: 'Submitted', show: true, min_width: 100},
        ];

    });
    app.controller("ds_controller", function($scope, $controller) {
        $scope.title = 'Datasets';
        $scope.fields = [{name: 'Name', show: true, min_width: 150},
                         {name: 'refs', show: false},
                         {name: 'extension', show: false, min_width: 150},
                         {name: 'Backend', show: true, min_width: 150},
                         {name: 'Status', show: true, min_width: 150},
                         {name: 'elapsed', show: false},
                         {name: 'Submitted', show: true, min_width: 150}];
    });

    app.controller("all_jobs_controller", function($scope, $http) {
        $scope.jobs = [];

        $scope.load_jobs = function() {
            $http({
                method: 'GET',
                url: '/completed_jobs.json',
            }).then(function success(response) {
                var r = response.data;
                $scope.jobs = [].concat(r.running, r.datasets, r.models, r.evaluations);
            }, function error(response) {
                console.log(response.statusText);
            });
        };
        $scope.load_jobs();

        $scope.is_running = function(job) {
            return (job &&
                    (job.status == 'Initialized' ||
                     job.status == 'Waiting' ||
                     job.status == 'Running'));
        };

        $scope.not_running = function(job) {
            return (job &&
                    (job.status != 'Initialized' &&
                     job.status != 'Waiting' &&
                     job.status != 'Running'));
        };
        $scope.is_dataset = function(job) {
            return (job &&
                    (!$scope.is_running(job) &&
                     job.type == 'dataset'));
        };

        $scope.is_model = function(job) {
            return (job.type == 'model');
        };

        $scope.is_evaluation = function(job) {
            return (job.type == 'evaluation');
        };
        $scope.set_attribute = function (job_id, name, value) {
            for (var i = 0; i < $scope.jobs.length; i ++) {
                if ($scope.jobs[i].id == job_id) {
                    $scope.jobs[i][name] = value;
                    return true;
                }
            }
            return false;
        }
    });

    app.controller('job_controller', function($scope, $controller) {
        $scope.column = 'id';
        $scope.reverse = false;
        $scope.change_sorting = function(parameter, event) {
            $scope.column = parameter;
            if ($scope.reverse) {
                $scope.reverse = false;
            } else {
                $scope.reverse = true;
            }
        };

        $scope.print_time_diff_ago = function(start) {
            return print_time_diff_ago(start, 'minute');
        };

        $scope.print_time_diff_simple = function(diff) {
            return print_time_diff_simple(diff);
        };

        $scope.print_time_diff_terse = function(diff) {
            return print_time_diff_terse(diff);
        };

        $scope.print_time_diff = function(diff) {
            return print_time_diff(diff);
    };

    });

    app.directive('dgName', function() {
        return {
            restrict: 'AE',
            replace: true,
            template: (
                        '    <a href="' + URL_PREFIX + '/jobs/{[ job.id ]}" title="{[job.name]}">' +
                        '        {[ job.name ]}' +
                        '    </a>'),
        };
    });

    app.directive('modelName', function () {
        return {
            restrict: 'AE',
            replace: true,
            template: (
                        '    <a href="' + URL_PREFIX + '/jobs/{[ job.model.id ]}" title="{[job.model.name]}">' +
                        '        {[ job.model.name ]}' +
                        '    </a>'),
        };
    });
    app.directive('datasetName', function () {
        return {
            restrict: 'AE',
            replace: true,
            template: (
                        '    <a href="' + URL_PREFIX + '/jobs/{[ job.dataset.id ]}" title="{[job.dataset.name]}">' +
                        '        {[ job.dataset.name ]}' +
                        '    </a>'),
        };
    });

    // Because jinja uses {{ and }}, tell angular to use {[ and ]}
    app.config(['$interpolateProvider', function($interpolateProvider) {
        $interpolateProvider.startSymbol('{[');
        $interpolateProvider.endSymbol(']}');
    }]);
})();
}
catch (ex) {
    console.log(ex);
}

$(document).ready(function(){
    if (typeof(socket) !== 'undefined') {
        socket.on('task update', function(msg) {
             console.log('socketio task update' + msg);
        });
        socket.on('job update', function(msg) {
            console.log('socketio job update' + msg);
            var scope = angular.element(document.getElementById('all-jobs')).scope();
            if (msg.update == 'status') {
                if (scope.set_attribute(msg.job_id, 'status', msg.status)
                    && scope.set_attribute(msg.job_id, 'status_css', msg.css)) {
                    scope.$apply()
                }

            } else if (msg.update == 'progress') {
                if (scope.set_attribute(msg.job_id, 'progress', msg.percentage)) {
                    scope.$apply();
                }
            } else if (msg.update == 'added') {
            } else if (msg.update == 'deleted') {
            } else if (msg.update == 'attribute') {
                if (scope.set_attribute(msg.job_id, msg.attribute, msg.value)) {
                    scope.$apply();
                }
            }

        });
    }
});
